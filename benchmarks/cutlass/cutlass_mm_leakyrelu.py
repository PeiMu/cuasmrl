import argparse

import triton
import triton.language as tl

import torch
import numpy as np

import cutlass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', action='store_true')
    parser.add_argument('-m', type=int, default=512)
    parser.add_argument('-n', type=int, default=512)
    parser.add_argument('-k', type=int, default=2048)
    args = parser.parse_args()
    return args

def get_cuda_autotune_config():
    return [
            # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1,
            #               num_warps=8),
            # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
            #               num_warps=4),
            # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
            #               num_warps=4),
            # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
            #               num_warps=4),
            # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
            #               num_warps=4),
            # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
            #               num_warps=4),
            # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
            #               num_warps=2),
            # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
            #               num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                        num_warps=2),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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
        accumulator = tl.dot(a, b, accumulator)
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

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel = matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION='leaky_relu'  #
    )
    return c



def main():
    args = parse_args()
    m, n, k = args.m, args.n, args.k
    plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
    # plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.ColumnMajor)

    # print avialable act
    print('===============================')
    print('list of epilogue')
    activations = plan.activations()
    for activation in activations:
        print(activation)
    print('===============================')

    # NOTE: This controls whether ther C++ GEMM declaration will be printed at each step. Set to `false` to
    # omit this information.
    print_module = False

    type_A = np.float16
    type_B = np.float16
    type_C = np.float16
    type_D = np.float16

    np.random.seed(1234)
    scope_min = -4
    scope_max = 4
    tensor_A = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, k)).astype(type_A))
    tensor_B = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(k, n)).astype(type_B))
    tensor_C = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, n)).astype(type_C))

    alpha = np.float16(1.)
    beta = np.float16(0.)

    tensor_D = np.zeros(tensor_C.shape).astype(type_D)
    tensor_D_relu = np.zeros(tensor_C.shape).astype(type_D)

    # test relu
    # plan.activation = "relu"
    # plan.run(tensor_A, tensor_B, tensor_C, tensor_D_relu, print_module=print_module)
    # relu_ref = (tensor_D >= 0).astype(type_D) * tensor_D
    # np.testing.assert_array_equal(relu_ref, tensor_D_relu)


    # leaky_relu
    negative_slope = 0.01
    plan.activation = ("leaky_relu", negative_slope)
    plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module)

    if not args.b:
        return 

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            #x_vals=[2 ** i for i in range(8, 13)],  # Different possible values for `x_name`
            x_vals=[0],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["triton", "cutlass"],
            line_names=['triton', 'cutlass'],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            "fp16",  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": None},
        ))


    @triton.testing.perf_report(configs)
    def benchmark(M, provider, fp8_inputs):
        M, K, N = args.m, args.k, args.n
        # llama matmul: [seqLen,4096] x [4096,4096]
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cutlass':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module), quantiles=quantiles, warmup=100, rep=100)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles, warmup=100, rep=100)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    
    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    main()
