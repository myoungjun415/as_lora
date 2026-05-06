import argparse

import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--lr_a", type=float, default=0.25)
    parser.add_argument("--lr_b", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./logs")
    return parser.parse_args()

def main():
    args = parse_args()
    module_name = f"glue.{args.task}.{args.method}"
    module = importlib.import_module(module_name)
    module.run(args)

if __name__ == "__main__":
    main()