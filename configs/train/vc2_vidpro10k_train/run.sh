export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=4
current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="timestep_strategy1"
CONFIG='configs/train/vc2_vidpro10k_train/config.yaml' # experiment config 
LOGDIR="./results/dpo-vc2-12th"                         # experiment saving directory all should under subfolder so that won't be copied to codeversion


# 等这么久之后，生成图片差不多完了
# echo "Script starts..."
# sleep 5h
# echo "5 hours later..."

# ### run
# python scripts/train.py \
# -t --devices '4,' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True \
# --gput_num 4


python -m torch.distributed.run \
--nnodes=1 \
--nproc_per_node=4 \
scripts/train.py \
-t --devices '0,1,2,3' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$EXPNAME"_"$current_time" \
--logdir $LOGDIR \
--auto_resume True

# --master_port=29501 \
# --name "$current_time"_$EXPNAME \