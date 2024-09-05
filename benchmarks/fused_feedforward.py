import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

from cuasmrl.jit import jit
from cuasmrl.autotuner import autotune
from cuasmrl.utils.gpu_utils import get_gpu_name, get_gpu_cc

from cuasmrl.autotuner import triton_autotune_with_cache

# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: bool = False
    tt: bool = False

    # Workload
    b: int = 1
    m: int = 1
    n: int = 1
    k: int = 1

    # RL
    train: int = 1
    log: int = 1
    verbose: int = 0
    ## Env
    env_id: str = 'cuasmenv-v0'
    num_env: int = 1
    num_iterations: int = 200
    minibatch_size: int = 8
    horizon: int = 32
    num_steps: int = 64
    normalize_reward: int = 0
    ckpt_freq: int = 100
    ## Agent
    agent: str = "ppo"
    weights_path: Optional[str] = None
    agent_id: Optional[str] = None
    anneal_lr: int = 1
    gae: int = 1
    norm_adv: int = 1
    clip_vloss: int = 1
    update_epochs: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    gpu: int = 0

def parse_args():
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("-b", type=int, default=1)
    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=1024)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(200))
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, dest="horizon", default=32)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--normalize_reward", type=int, default=0)
    parser.add_argument("--ckpt_freq", type=int, default=100)

    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--agent_id", type=str)
    parser.add_argument("--anneal_lr", type=int, default=1)
    parser.add_argument("--gae", type=int, default=1)
    parser.add_argument("--norm_adv", type=int, default=1)
    parser.add_argument("--clip_vloss", type=int, default=1)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = Config(**vars(args))
    return config

GPU = get_gpu_name()

# BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
# num_stages=2, num_warps=4
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M':8 }, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=5, num_warps=2),

        # origin
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M':8 }, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tt_ff(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    # native
    # pid = tl.program_id(axis=0)
    # pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    # pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # L2
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    ##################

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)

        # a_sum += tl.math.pow(a.to(tl.float32), 2)
        a_float32 = a.to(tl.float32)
        a_sum += a_float32 * a_float32

        rms_w = tl.load(rms_w_ptrs)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float32)
            b = b.to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True)
            c = c.to(tl.float32)
            c = c.to(tl.float16)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w

    a_mean = tl.sum(a_sum, axis=1) / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def call_tt(M, N, K, x: torch.Tensor, x_reshape, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor, out, grid) -> torch.Tensor:
    # assert x.dtype == torch.float16
    # assert w1.dtype == w3.dtype == rms_w.dtype
    # assert w1.dtype in [torch.int8, torch.float16]
    # assert w1.shape == w3.shape

    w1_t = w1.t()
    w3_t = w3.t()

    tt_ff[grid](
        x_reshape, w1_t, w3_t, out, rms_w,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        *rms_w.stride(),
        USE_FP8=False,
        EPS=1e-6,
        # BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
        # num_stages=2, num_warps=4
    )
    # out = out.view(batch, seq_len, -1)
    return out

def call(M, N, K, x, x_reshape, w1, w3, rms_w,  out, grid, kernel, load_dir):
    w1_t = w1.t()
    w3_t = w3.t()
    kernel[grid](
        x_reshape, w1_t, w3_t, out, rms_w,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        *rms_w.stride(),
        USE_FP8=False,
        EPS=1e-6,

        # cuasmrl
        load_dir=load_dir,
    )
    return out


def rms_norm_pytorch(x: torch.Tensor, rms_w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_w


def ff_pytorch(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    x_norm = rms_norm_pytorch(x, rms_w, eps=1e-6)
    a = torch.nn.functional.silu(torch.matmul(x_norm, w1.t()))
    b = torch.matmul(x_norm, w3.t())
    return a * b

def main():
    drl_config = parse_args()
    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    B, M, K, N = drl_config.b, drl_config.m, drl_config.k, drl_config.n
    x = torch.randn([B, M, K], dtype=torch.float16, device="cuda")
    x_reshape = x.reshape(B*M, K)
    # weights tends to be very small values
    rms_w = torch.randn([K], dtype=torch.float16, device="cuda") * 0.2
    w1_w = torch.randn([N, K], dtype=torch.float16, device="cuda") * 0.2
    w3_w = torch.randn([N, K], dtype=torch.float16, device="cuda") * 0.2

    drl_config.total_flops = B*M*N*K*2
    drl_config.save_dir = f'{GPU}/fused_ff/{B}_{M}_{N}_{K}'

    if drl_config.load is None:
        load_dir = None
    elif drl_config.load == "auto":
        load_dir = f'{drl_config.default_out_path}/{GPU}/fused_ff/{B}_{M}_{N}_{K}'
    else:
        load_dir = drl_config.load

    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)

    @autotune(
        configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M':8 }, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M':8 }, num_stages=5, num_warps=2),

        # origin
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M':8 }, num_stages=2, num_warps=4),
        ],
        key=['M', 'N', 'K'],
        ret_ptr=3,
        drl_config=drl_config,
    )
    @jit
    def cuasmrl_kernel(
        a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_w1k, stride_w1n,
        stride_w3k, stride_w3n,
        stride_outm, stride_outn,
        stride_rms_w,
        USE_FP8: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        w1 and w3 are weights (linear layers)
        F.silu(w1(x)) * w3(x)
        """
        # native
        # pid = tl.program_id(axis=0)
        # pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
        # pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

        # L2
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        ##################

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
        w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
        a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)

            # a_sum += tl.math.pow(a.to(tl.float32), 2)
            a_float32 = a.to(tl.float32)
            a_sum += a_float32 * a_float32

            rms_w = tl.load(rms_w_ptrs)
            if USE_FP8:
                rms_w = rms_w.to(tl.float8e5, bitcast=True)
                rms_w = rms_w.to(tl.float16)
            a = a * rms_w
            b = tl.load(w1_ptrs)
            if USE_FP8:
                b = b.to(tl.float8e5, bitcast=True)
                b = b.to(tl.float32)
                b = b.to(tl.float16)
            acc1 += tl.dot(a, b)
            c = tl.load(w3_ptrs)
            if USE_FP8:
                c = c.to(tl.float8e5, bitcast=True)
                c = c.to(tl.float32)
                c = c.to(tl.float16)
            acc2 += tl.dot(a, c)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            w1_ptrs += BLOCK_SIZE_K * stride_w1k
            w3_ptrs += BLOCK_SIZE_K * stride_w3k

            rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w

        a_mean = tl.sum(a_sum, axis=1) / K + EPS
        a_norm = tl.math.rsqrt(a_mean)
        acc1 = acc1 * a_norm[:, None]
        acc2 = acc2 * a_norm[:, None]
        accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

        offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
        out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
        tl.store(out_ptrs, accumulator, mask=out_mask)

    o = torch.empty((B*M, N), dtype=x.dtype, device=x.device)
    out = call(B*M, N, K, x, x_reshape, w1_w, w3_w, rms_w, o, grid, cuasmrl_kernel, load_dir)

    if drl_config.tt:
        o = torch.empty((B*M, N), dtype=x.dtype, device=x.device)
        output_triton = call_tt(B*M, N, K, x, x_reshape, w1_w, w3_w, rms_w, o, grid)
        output_pytorch = ff_pytorch(x=x, w1=w1_w, w3=w3_w, rms_w=rms_w)
        output_triton = output_triton.view(B, M, -1)
        assert torch.allclose(output_triton, output_pytorch, atol=5e-1), f"max diff: {torch.max(torch.abs(output_triton - output_pytorch))}"

        # print("rms matmul silu mul triton", triton.testing.do_bench(lambda: call_tt(x, x_reshape, w1_w, w3_w, rms_w, o, grid)))
        # print("rms matmul silu mul pytorch", triton.testing.do_bench(lambda: ff_pytorch(x=x, w1=w1_w, w3=w3_w, rms_w=rms_w)))

    # w1_w_fp8 = f16_to_f8(w1_w, dtypes=tl.float8e5)
    # w3_w_fp8 = f16_to_f8(w3_w, dtypes=tl.float8e5)
    # rms_w_fp8 = f16_to_f8(rms_w, dtypes=tl.float8e5)

    # out_fp8 = kernel_ff(x=x, w1=w1_w_fp8, w3=w3_w_fp8, rms_w=rms_w_fp8)
    # # on very large tensors, it is expected that the error is large, we just check it is not crazy large
    # assert torch.allclose(out_fp8, w1_silu_p * w3_p, atol=10)
    #
    # print("rms matmul silu mul triton fp8", triton.testing.do_bench(lambda: kernel_ff(x=x, w1=w1_w_fp8, w3=w3_w_fp8, rms_w=rms_w_fp8)))

    if not drl_config.bench:
        return

    assert drl_config.load is not None
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["NA"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            #x_vals=[2 ** i for i in range(8, 13)],  # Different possible values for `x_name`
            x_vals=[0],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot

            line_vals=["triton", "torch", 'cuasmrl'],
            line_names=['triton', 'torch', 'cuasmrl'],

            #line_vals=["torch", "triton", ],
            #line_names=[ 'torch','triton', ],

            styles=[("green", "-"), ("blue", "-"), ('red', '-')],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" + "fp16",
            args={"fp8_inputs": None},
        ))

    @triton.testing.perf_report(configs)
    def benchmark(NA, provider, fp8_inputs):
        o = torch.empty((B*M, N), dtype=x.dtype, device=x.device)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ff_pytorch(x=x, w1=w1_w, w3=w3_w, rms_w=rms_w) ,warmup=100, rep=100,  quantiles=quantiles)
        if provider == 'cuasmrl':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: call(B*M, N, K, x, x_reshape, w1_w, w3_w, rms_w, o, grid, cuasmrl_kernel, load_dir), quantiles=quantiles, warmup=100, rep=100)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: call_tt(B*M, N, K, x, x_reshape, w1_w, w3_w, rms_w, o, grid), quantiles=quantiles, warmup=100, rep=100)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    main()
