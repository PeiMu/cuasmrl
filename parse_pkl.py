from cuasmrl.utils.record import save_data, read_data

for id in range(8):
  target_file = f'/root/cuasmrl/results/512_512_2048/cuasmrl_kernel_0d1d2_{id}.pkl'

  data = read_data(target_file)
  sass = data['sass']
  cubin = data['cubin']
  final_perf = data['final_perf']
  init_perf = data['init_perf']

  with open(f'./mm_leakyRelu_{id}.sass', 'w') as f:
      f.write(sass)
      f.close()

  with open(f'./mm_leakyRelu_{id}.cubin', 'wb') as f:
      f.write(cubin)
      f.close()

  print(f"save mm_leakyRelu_{id} sass and cubin with final_perf={final_perf}, init_perf={init_perf}")
