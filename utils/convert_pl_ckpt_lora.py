import torch
import argparse

def convert_ckpt(input_path, output_path):
    ckpt = torch.load(input_path)['state_dict']
    torch.save(ckpt, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint.")
    parser.add_argument("--input_path", type=str, help="Path to the input checkpoint file.")
    parser.add_argument("--output_path", type=str, help="Path to save the converted checkpoint.")

    args = parser.parse_args()

    convert_ckpt(args.input_path, args.output_path)
