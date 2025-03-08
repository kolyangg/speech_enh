import torch
import os

def analyze_checkpoint(file_path):
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        num_tensors = len(state_dict)
        total_params = sum(p.numel() for p in state_dict.values())

        file_size_mb = os.path.getsize(file_path) / (1024 ** 2)

        print(f"✅ Loaded '{file_path}' successfully.")
        print(f"- File size on disk: {file_size(file_path):.2f} MB")
        print(f"- Number of tensors in state_dict: {num_tensors}")
        print(f"- Total parameters in state_dict: {total_params:,}")

        if 'ema' in checkpoint:
            ema_data = checkpoint['ema']
            if isinstance(ema_data, dict) and 'state_dict' in ema_data:
                ema_state_dict = ema_data['state_dict']
                ema_num_tensors = len(ema_state_dict)
                ema_total_params = sum(p.numel() for p in ema_state_dict.values())

                print("EMA is present:")
                print(f"- Number of tensors in EMA state_dict: {ema_num_tensors}")
                print(f"- Total parameters in EMA state_dict: {ema_total_params:,}")
            else:
                print("EMA is present but does not contain a valid 'state_dict'.")
        else:
            print("EMA is not present.")

        print("- Other keys in checkpoint:", list(checkpoint.keys()))
        print()

    except Exception as e:
        print(f"❌ Failed to load '{file_path}'. Error: {e}\n")

def file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

files = [
    './exp/universepp_vb_16k/2025-02-28_11-58-42_/checkpoints/epoch=0.ckpt',
    './exp/universepp_vb_16k/2025-03-06_23-30-48_/checkpoints/universe/exper/last.ckpt'
]

for file in files:
    analyze_checkpoint(file)
