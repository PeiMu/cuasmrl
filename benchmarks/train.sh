
#! /bin/bash
mkdir data

python3 benchmarks/03-matrix-multiplication.py
python3 benchmarks/batch_matmul.py
python3 benchmarks/fused_feedforward.py
python3 benchmarks/02-fused-softmax.py
python3 benchmarks/rmsnorm.py
python3 benchmarks/06-fused-attention.py
