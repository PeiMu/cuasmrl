import os
import argparse
from dataclasses import dataclass, field

from cuasmrl.backend import MutationEngine
from functools import partial

# yapf: disable
@dataclass
class Config:
    path: str = "data"
    end: int = 5


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', type=str, dest="path", default="data")
    parser.add_argument('-e', type=int, dest='end', default=5)
    args = parser.parse_args()
    config = Config(**vars(args))
    return config

decode = partial(MutationEngine.decode, None)
decode_ctrl_code = partial(MutationEngine.decode_ctrl_code, None)

def main():

    config = parse_args()

    with open(config.path, "r") as f:
        for i, line in enumerate(f):
            # print(line.rstrip())

            ctrl_code, comment, predicate, opcode, dest, src, meta = decode(line)

            print('line is ', line)
            print('decoding: ')
            print(ctrl_code)
            if ctrl_code is not None:
                # skip label
                print(decode_ctrl_code(ctrl_code))
            print(predicate)
            print(opcode)
            print(dest)
            print(src)
            print(meta)
            print()

            if i > config.end:
                break

if __name__ == "__main__":
    main()
