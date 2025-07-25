from cuasmrl.utils.record import save_data, read_data

target_file = '/home/pei/Project/cuasmrl/data/NVIDIA_A100-SXM4-80GB/mm_leakyRelu/512_512_2048/cuasmrl_kernel_0d1d2_0.pkl'

data = read_data(target_file)
print("success")
