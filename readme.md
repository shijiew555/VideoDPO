# checkpoints

save checkpoints to `checkpoints/vc2/model.ckpt`
use `python utils/create_ref_model.py` to create `ref_model.ckpt`


# Install

Already adjust requirements.txt to H800.
```shell
conda create -n videocrafter-dpo python=3.8
pip install -r requirements.txt
```
# Functions
## Finetune videocrafter
(1) direct run:
```
# apply 2 gpus for debug run
srun -p project --gres=gpu:2 --pty $SHELL 
conda activate videocrafter
bash configs/train/000_videocrafter2ft/run.sh
```
(2) submit a job:
```
sbatch configs/train/  000_videocrafter2ft/run_slurm.sh
```
