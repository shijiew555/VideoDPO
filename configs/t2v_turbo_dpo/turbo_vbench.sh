#!/bin/bash

convert_and_infer() {
    local exp_root="results/exp_round/"
    local exp_name="exp_name"
    local ckpt_path="/path/to/checkpoint.ckpt"
    for ckpt_file in "$ckpt_dir"/*.ckpt; do
        # 确保文件存在
        if [ -f "$ckpt_file" ]; then
            # 获取文件名
            local ckpt_name=$(basename "$ckpt_file")

            # 转换检查点
            echo "Converting $ckpt_name..."
            python utils/convert_pl_ckpt_lora.py --input_path "$ckpt_file" --output_path "$ckpt_file"
            
            # 执行 turbo inference
            echo "Running inference on $ckpt_name..."
            python scripts/turbo_inference/text2video.py \
                --unet_dir "$ckpt_file" \
                --base_model_dir /root/init_ckpt/vc2/model.ckpt \
                --prompts_file /root/VideoDPOData/prompts/vbench_standard_all.txt \
                --output_dir "/root/autodl-tmp/vbench_results/${exp_name}_${ckpt_name%.ckpt}"
        else
            echo "No .ckpt files found in $checkpoint_dir"
        fi
    done
}

# 调用函数
convert_and_infer
