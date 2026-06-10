"""
engines/sglang_backend.py

SGLang server launcher + custom Triton attention backend registration.

SGLang uses RadixAttention for KV-cache sharing and chunked prefill.
This module:
  1. Launches SGLang server with stock FlashAttention-2 (baseline)
  2. Launches SGLang server with custom Triton kernel (experimental)
  3. Provides the CustomAttnBackend class for SGLang's attention dispatch
"""

import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path
from typing import Optional


# ── SGLang custom attention hook ─────────────────────────────────────────────

try:
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from kernels.triton_attention import TritonAttention

    class SGLangTritonBackend:
        """
        Custom attention backend for SGLang.

        SGLang dispatches attention through sglang.srt.layers.attention.
        Subclassing here allows injecting the Triton kernel while keeping
        RadixAttention's KV-cache sharing logic intact.

        Register via:
          from engines.sglang_backend import SGLangTritonBackend
          import sglang.srt.layers.attention as attn_module
          attn_module._CUSTOM_BACKEND = SGLangTritonBackend(...)
        """

        def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
            self.kernel = TritonAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                causal=True,
            )

        def __call__(
            self,
            q: "torch.Tensor",
            k: "torch.Tensor",
            v: "torch.Tensor",
            **kwargs,
        ) -> "torch.Tensor":
            """
            SGLang passes (num_tokens, num_heads, head_dim) tensors.
            Reshape → BHSD → kernel → reshape back.
            """
            # treat full batch as single-sequence batch
            q_4d = q.unsqueeze(0).transpose(1, 2)   # (1, H, T, D)
            k_4d = k.unsqueeze(0).transpose(1, 2)
            v_4d = v.unsqueeze(0).transpose(1, 2)
            out  = self.kernel(q_4d, k_4d, v_4d)
            return out.transpose(1, 2).squeeze(0)    # (T, H, D)

except ImportError:
    pass


# ── Server launcher ───────────────────────────────────────────────────────────

def launch_sglang_server(
    model_path: str,
    port: int = 30000,
    tensor_parallel: int = 2,
    mem_fraction_static: float = 0.88,
    dtype: str = "float16",
    max_prefill_tokens: int = 8192,
    attention_backend: str = "flashinfer",   # "flashinfer" | "triton"
    log_file: Optional[str] = None,
) -> subprocess.Popen:
    """
    Launches SGLang OpenAI-compatible server.

    attention_backend:
      "flashinfer"  → SGLang default (FlashInfer CUDA, RadixAttention)
      "triton"      → custom Triton kernel via env-var hook
    """
    env = os.environ.copy()

    if attention_backend == "triton":
        # SGLang reads this env var to load a custom attention module path
        env["SGLANG_CUSTOM_ATTENTION_BACKEND"] = str(
            Path(__file__).parent / "sglang_triton_hook.py"
        )

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path",           model_path,
        "--port",                 str(port),
        "--tp",                   str(tensor_parallel),
        "--mem-fraction-static",  str(mem_fraction_static),
        "--dtype",                dtype,
        "--max-prefill-tokens",   str(max_prefill_tokens),
        "--chunked-prefill-size", "512",   # SGLang-specific: chunked prefill
        "--disable-radix-cache" if attention_backend == "triton" else "",
    ]
    # remove empty strings
    cmd = [c for c in cmd if c]

    print(f"Launching SGLang [{attention_backend}] on port {port}...")
    stdout = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc   = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)
    _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=300)
    print(f"SGLang [{attention_backend}] ready on port {port}")
    return proc


def stop_server(proc: subprocess.Popen, timeout: int = 30):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_server(url: str, timeout: int = 300, interval: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")
