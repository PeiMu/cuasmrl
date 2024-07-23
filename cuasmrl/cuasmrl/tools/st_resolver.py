import argparse
from functools import partial
from cuasmrl.backend import MutationEngine

def parse_args():
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("-m", type=int, default=512)

    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    return args

decode = partial(MutationEngine.decode, None)
ST_MAP = {}

def main():
    args = parse_args()

    # read sass file

    # for each kernel

    # for each inst

    # find its def

    decode


if __name__ == '__main__':
    main()