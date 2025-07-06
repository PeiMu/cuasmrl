
#! /bin/bash

for i in {1..5}; do python3 benchmarks/cutlass/cutlass_mm_leakyrelu.py --load auto --bench; done
for i in {1..5}; do python3 benchmarks/batch_matmul.py --load auto --bench; done
for i in {1..5}; do python3 benchmarks/fused_feedforward.py --load auto --bench; done
for i in {1..5}; do python3 benchmarks/02-fused-softmax.py --load auto --bench; done
for i in {1..5}; do python3 benchmarks/cutlass/torch_rmsnorm.py --load auto --bench; done
for i in {1..5}; do python3 benchmarks/06-fused-attention.py --load auto --bench; done

