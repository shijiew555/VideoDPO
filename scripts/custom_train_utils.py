import yaml
import json
import os

def create_train_data_yaml(dataset_list, dst_yaml_path):
    data_config = {
        'META': dataset_list
    }
    with open(dst_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

def create_config_yaml(data_root_yaml, src_cfg_path, dst_cfg_path):
    with open(src_cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 假设config中有一个data参数，我们更新它的data_root字段
    if 'data' in config and 'params' in config['data']:
        config['data']['params']['train']['params']['data_root'] = data_root_yaml
    else:
        raise KeyError("Source configuration does not contain 'data' or 'params' keys")

    with open(dst_cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def construct_cmd(expname,
                 config,
                 logdir):
    # default_expname="sub_expname_of_this_round"
    # default_config='configs/train/vc2_vidpro10k_train/config.yaml'
    # default_logdir="./results/batch_train_name"
    cmd = f"bash configs/train/driver/run.sh {expname} {config} {logdir}"
    print(cmd)

# 设置环境变量
# 使用示例
data_folder = "/home/rliuay/haoyu/dataset/text2video2-10k/pair_policy"
dataset_list = os.listdir(data_folder)
dataset_lists = [[os.path.join(data_folder,i)] for i in dataset_list]
src_cfg_path = '/home/rliuay/haoyu/research/DPO-videocrafter/configs/train/driver/config.yaml'

for dataset_list in dataset_lists:
    expname = os.path.basename(dataset_list[0])#total_score_0P_pair01_vbscore
    rfd = f"/home/rliuay/haoyu/research/DPO-videocrafter/configs/train/driver/train_configs/{expname}"
    os.makedirs(rfd,exist_ok=True)
    dst_yaml_path = f'{rfd}/train_data.yaml'
    dst_cfg_path  = f'{rfd}/config.yaml'
    create_train_data_yaml(dataset_list, dst_yaml_path)
    create_config_yaml(dst_yaml_path, src_cfg_path, dst_cfg_path)
    construct_cmd(expname,
                  dst_cfg_path,
                 "./results/dpo-vc2-10th")