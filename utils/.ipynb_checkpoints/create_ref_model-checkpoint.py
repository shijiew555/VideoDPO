"""
haoyu
20240602
create a ref_model from ckpt 
now support I2V model
"""

import torch 
import os

def print_file_size_in_gb(file_path):
    if os.path.isfile(file_path):
        file_size_bytes = os.path.getsize(file_path)
        file_size_gb = file_size_bytes / (1024 ** 3)
        print(f"The size of the file '{file_path}' is: {file_size_gb:.2f} GB")
    else:
        print(f"The file '{file_path}' does not exist.")


def videocrafter2refmodel(videocrafter_path,ref_model_path):
    ckpt = torch.load(videocrafter_path)['state_dict']
    new_state_dict = {}
    diffusion_key = "model.diffusion_model"
    target_prefix = "ref_model.diffusion_model"
    for k,v in ckpt.items():
        if k.startswith(diffusion_key):
            newk = k.replace(diffusion_key,target_prefix)
            new_state_dict[newk] = ckpt[k]

    torch.save({'state_dict':new_state_dict},ref_model_path)


if __name__=="__main__":
    path = "/home/rliuay/haoyu/research/LVDM-UST-VideoCrafterft-master/init_ckpt/VideoCrafter1-Text2Video-512/model.ckpt"
    target_path = "/home/rliuay/haoyu/research/LVDM-UST-VideoCrafterft-master/init_ckpt/VideoCrafter1-Text2Video-512/ref_model.ckpt"
    videocrafter2refmodel(path,target_path)
    ckpt = torch.load(target_path)['state_dict']
    # print(ckpt.keys())
    print_file_size_in_gb(target_path)
    