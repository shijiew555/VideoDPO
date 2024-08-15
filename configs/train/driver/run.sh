default_expname="sub_expname_of_this_round"
default_config='configs/train/vc2_vidpro10k_train/config.yaml'
default_logdir="./results/batch_train_name"
default_current_time=$(date +%Y%m%d%H%M%S)

# 如果传入了参数，使用传入的参数；否则使用默认值
expname=${1:-$default_expname}
config=${2:-$default_config}
logdir=${3:-$default_logdir}
current_time=${4:-$default_current_time}

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3


# 打印参数以供验证
echo "Exporting TOKENIZERS_PARALLELISM=false"
echo "Current time: $current_time"
echo "Experiment name: $expname"
echo "Experiment config: $config"
echo "Experiment saving directory: $logdir"


# 封装 Python 命令的 Bash 函数
run_experiment() {
    python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=4 \
        --master_port=29501 \
        scripts/train.py \
        -t --devices '0,1,2,3' \
        lightning.trainer.num_nodes=1 \
        --base "$config" \
        --name "${expname}_${current_time}" \
        --logdir "$logdir" \
        --auto_resume True
}

run_experiment
