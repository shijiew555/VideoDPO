#!/bin/bash

convert_and_infer() {
    local exp_root="results/exp_round/"
    local exp_name="exp_name"
    local ckpt_path="/path/to/checkpoint.ckpt"
    echo "Converting $ckpt_name..."
    python utils/convert_pl_ckpt_lora.py --input_path "$ckpt_path" --output_path "$ckpt_path"
    
    # 执行 turbo inference
    echo "Running inference on $ckpt_name..."
    python scripts/turbo_inference/text2video.py \
        --unet_dir "$ckpt_path" \
        --base_model_dir /root/init_ckpt/vc2/model.ckpt \
        --prompts_file /root/DPO-videocrafter/prompts/cvpr_demo_supp_vidpro10k-2.txt \
        --output_dir "vis_results/${exp_name}_${ckpt_name%.ckpt}"
}

# 调用函数
convert_and_infer
