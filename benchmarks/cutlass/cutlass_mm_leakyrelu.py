import argparse
from dataclasses import dataclass, field
from typing import Optional

import os
import random
import numpy as np
import torch

import triton
import triton.language as tl

from cuasmrl.jit import jit
from cuasmrl.autotuner import autotune as fgk_autotune
from cuasmrl.utils.gpu_utils import get_gpu_name

from cuasmrl.autotuner import triton_autotune_with_cache

import cutlass

# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: bool = False

    # Workload
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
    num_iterations: int = int(1e3)
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


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=2048)

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

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, c, kernel, M, N, K, grid, load, activation=""):
    kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,  #

        #
        load_dir=load,
    )
    return c

def torch_matmul(a, b):
    return torch.nn.functional.leaky_relu(torch.matmul(a, b)+1, negative_slope=0.01)



def main():
    args = parse_args()
    M, N, K = args.m, args.n, args.k
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    M, N, K = args.m, args.n, args.k
    args.total_flops = M*N*2*K
    args.save_dir=f'{GPU}/mm_leakyRelu/{M}_{N}_{K}'

    @fgk_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
        ],
        key=['M', 'N', 'K'],
        ret_ptr=2,
        drl_config=args,
    )
    @jit
    def cuasmrl_kernel(
            # Pointers to matrices
            a_ptr, b_ptr, c_ptr,
            # Matrix dimensions
            M, N, K,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            # by to get the element one row down (A has M rows).
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,
            # Meta-parameters
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
            GROUP_SIZE_M: tl.constexpr,  #
            ACTIVATION: tl.constexpr  #
    ):
        # -----------------------------------------------------------
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


    # @triton.autotune(
    @triton_autotune_with_cache(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
        ],
        key=['M', 'N', 'K'],
        drl_config=args,  # just need the path really
    )
    @triton.jit
    def tt_kernel(
            # Pointers to matrices
            a_ptr, b_ptr, c_ptr,
            # Matrix dimensions
            M, N, K,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            # by to get the element one row down (A has M rows).
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,
            # Meta-parameters
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
            GROUP_SIZE_M: tl.constexpr,  #
            ACTIVATION: tl.constexpr  #
    ):
        # -----------------------------------------------------------
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def tt_matmul(a, b, c, M, N, K, grid, activation=""):
        tt_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            ACTIVATION=activation,  #
        )
        return c

    ############################## CUTLASS ##############################
    # plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
    # # plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.ColumnMajor)

    # # print avialable act
    # print('===============================')
    # print('list of epilogue')
    # activations = plan.activations()
    # for activation in activations:
    #     print(activation)
    # print('===============================')

    # # NOTE: This controls whether ther C++ GEMM declaration will be printed at each step. Set to `false` to
    # # omit this information.
    # print_module = False

    # type_A = np.float16
    # type_B = np.float16
    # type_C = np.float16
    # type_D = np.float16

    # np.random.seed(1234)
    # scope_min = -4
    # scope_max = 4
    # tensor_A = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(M, K)).astype(type_A))
    # tensor_B = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(K, N)).astype(type_B))
    # tensor_C = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(M, N)).astype(type_C))

    # alpha = np.float16(1.)
    # beta = np.float16(0.)

    # tensor_D = np.zeros(tensor_C.shape).astype(type_D)
    # tensor_D_relu = np.zeros(tensor_C.shape).astype(type_D)

    # # test relu
    # # plan.activation = "relu"
    # # plan.run(tensor_A, tensor_B, tensor_C, tensor_D_relu, print_module=print_module)
    # # relu_ref = (tensor_D >= 0).astype(type_D) * tensor_D
    # # np.testing.assert_array_equal(relu_ref, tensor_D_relu)

    # # leaky_relu
    # negative_slope = 0.01
    # plan.activation = ("leaky_relu", negative_slope)
    # plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module)
    ############################## CUTLASS ##############################

    if not args.bench:
        return 

    assert args.load is not None
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["NA"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            #x_vals=[2 ** i for i in range(8, 13)],  # Different possible values for `x_name`
            x_vals=[0],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot

            line_vals=["triton", "cutlass", 'cuasmrl'],
            line_names=['triton', 'cutlass', 'cuasmrl'],
            # line_vals=["triton", "cutlass", ],
            # line_names=['triton', 'cutlass', ],

            styles=[("green", "-"), ("blue", "-"), ('red', '-')],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" + "fp16",  
            args={"fp8_inputs": None},
        ))


    @triton.testing.perf_report(configs)
    def benchmark(NA, provider, fp8_inputs):
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        c_ref = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        if args.load is None:
            load_dir = None
        elif args.load == "auto":
            load_dir = f'{args.default_out_path}/{GPU}/mm_leakyRelu/{M}_{N}_{K}'
        else:
            load_dir = args.load


        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cutlass':
            # ms, min_ms, max_ms = triton.testing.do_bench(lambda: plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module), quantiles=quantiles, warmup=100, rep=100)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul(a, b) ,warmup=100, rep=100,  quantiles=quantiles)
        if provider == 'cuasmrl':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c, cuasmrl_kernel, M, N, K, grid, load_dir, "leaky_relu"), quantiles=quantiles, warmup=100, rep=100)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: tt_matmul(a, b, c_ref, M, N, K, grid, "leaky_relu"), quantiles=quantiles, warmup=100, rep=100)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    
    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    main()
