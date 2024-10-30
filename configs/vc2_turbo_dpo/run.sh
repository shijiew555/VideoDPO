#export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=4
current_time=$(date +%Y%m%d%H%M%S)
#current_time=20240722100141

EXPNAME="vc2_turbo_dpo"
CONFIG='configs/vc2_turbo_dpo/config.yaml' # experiment config 
LOGDIR="./results/baselines"  # experiment saving directory all should under subfolder so that won't be copied to codeversion


# ### run
python scripts/train.py \
-t --devices '0,1,2,3' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume True \
--gpu_num 4


# python -m torch.distributed.run \
# --nnodes=1 \
# --nproc_per_node=1 \
# scripts/train.py \
# -t --devices '0,1,2,3' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$EXPNAME"_"$current_time" \
# --logdir $LOGDIR \
# --auto_resume True

# --master_port=29501 \
# --name "$current_time"_$EXPNAME \
