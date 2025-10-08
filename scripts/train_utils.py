import argparse, os, sys, glob
import multiprocessing as mproc
from packaging import version
from omegaconf import OmegaConf
from collections import OrderedDict
import logging

mainlogger = logging.getLogger("mainlogger")

import torch
import pytorch_lightning as pl
from utils.load_weigths import load_from_pretrainedSD_checkpoint
from collections import OrderedDict


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")
    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if (
            "callbacks" in lightning_config
            and "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks
        ):
            os.makedirs(os.path.join(ckptdir, "trainstep_checkpoints"), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(
            OmegaConf.create({"lightning": lightning_config}),
            os.path.join(cfgdir, "lightning.yaml"),
        )
        # 2024 0728 haoyu
        # save data.yaml to the work sapace
        src_path = model_config['data']['params']['train']['params']['data_root'] ## xx. configs/data/vidpro/train_data.yaml
        root_dir = "/home/rliuay/haoyu/research/DPO-videocrafter"
        # get current workdir 
        
        config_path = os.path.join(root_dir,src_path)
        save_path = os.path.join(cfgdir,"train_data.yaml")
        os.system(f"cp {src_path} {save_path}")
        # import pdb;pdb.set_trace()
        # print("test saving train data .yaml exiting");exit();
        # save_path = os.path.join(cfgdir,"valid_data.yaml")
    # exit()
    return workdir, ckptdir, cfgdir, loginfo


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": True,
            },
        },
        "batch_logger": {
            "target": "utils.callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step", "log_momentum": False},
        },
        "cuda_callback": {"target": "utils.callbacks.CUDACallback"},
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks:
        mainlogger.info(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch}-{step}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 50,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg


def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg


def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPStrategy"
        # "target": "pytorch_lightning.strategies.DDPShardedStrategy"  # deprecated in newer versions
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg


def load_checkpoints(model, model_cfg):
    ## special load setting for dpo training
    ## modified on 20240521
    if check_config_attribute(model_cfg, "ref_model_checkpoint"):
        ### 0602 ： fix to load correct weights
        ### load videocrafter checkpoint
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        ref_model_checkpoint = model_cfg.ref_model_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(
            f">>> Load weights from pretrained checkpoint {pretrained_ckpt}"
        )
        # mainlogger.info(pretrained_ckpt)
        # print(f"Loading model from {pretrained_ckpt}")
        videocrafter_ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        ref_model_ckpt = torch.load(ref_model_checkpoint, map_location="cpu")[
            "state_dict"
        ]
        if "state_dict" in videocrafter_ckpt.keys():
            videocrafter_ckpt = videocrafter_ckpt["state_dict"]
        else:
            # deepspeed
            new_videocrafter_ckpt = OrderedDict()
            for key in videocrafter_ckpt["module"].keys():
                new_videocrafter_ckpt[key[16:]] = videocrafter_ckpt["module"][key]
            videocrafter_ckpt = new_videocrafter_ckpt

        videocrafter_state_dict = {**videocrafter_ckpt, **ref_model_ckpt}
        # model, empty_paras = get_empty_params_comparedwith_sd(model, model_cfg)
        model.load_state_dict(videocrafter_state_dict, strict=False)
        return model

    ## special load setting for adapter training
    if check_config_attribute(model_cfg, "adapter_only"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(
            f">>> Load weights from pretrained checkpoint (training adapter only) {pretrained_ckpt}"
        )
        # print(f"Loading model from {pretrained_ckpt}")
        ## only load weight for the backbone model (e.g. latent diffusion model)
        state_dict = torch.load(pretrained_ckpt, map_location=f"cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        else:
            # deepspeed
            dp_state_dict = OrderedDict()
            for key in state_dict["module"].keys():
                dp_state_dict[key[16:]] = state_dict["module"][key]
            state_dict = dp_state_dict
        model.load_state_dict(state_dict, strict=False)
        model.empty_paras = None
        return model
    empty_paras = None
    if check_config_attribute(model_cfg, "train_temporal"):
        mainlogger.info(">>> Load weights from [Stable Diffusion] checkpoint")
        model_pretrained, empty_paras = get_empty_params_comparedwith_sd(
            model, model_cfg
        )
        model = model_pretrained

    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(
            f">>> Load weights from pretrained checkpoint {pretrained_ckpt}"
        )
        # mainlogger.info(pretrained_ckpt)
        # print(f"Loading model from {pretrained_ckpt}")
        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")

        try:
            if "state_dict" in pl_sd.keys():
                model.load_state_dict(pl_sd["state_dict"])
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd["module"].keys():
                    new_pl_sd[key[16:]] = pl_sd["module"][key]
                model.load_state_dict(new_pl_sd)
        except:
            model.load_state_dict(pl_sd)
        """
        try:
            model = model.load_from_checkpoint(pretrained_ckpt, **model_cfg.params)
        except:
            mainlogger.info("[Warning] checkpoint NOT complete matched. To adapt by skipping ...")
            state_dict = torch.load(pretrained_ckpt, map_location=f"cpu")
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model_state_dict = model.state_dict()
            ## for layer with channel changed (e.g. GEN 1's conditon-concatenating setting)
            for n, p in model_state_dict.items():
                if p.shape != state_dict[n].shape:
                    mainlogger.info(f"Skipped parameter [{n}] from pretrained! ")
                    state_dict.pop(n)
            model_state_dict.update(state_dict)
            model.load_state_dict(model_state_dict)    
        empty_paras = None
        """
    elif check_config_attribute(model_cfg, "sd_checkpoint"):
        mainlogger.info(">>> Load weights from [Stable Diffusion] checkpoint")
        model_pretrained, empty_paras = get_empty_params_comparedwith_sd(
            model, model_cfg
        )
        model = model_pretrained

    else:
        empty_paras = None

    ## record empty params
    model.empty_paras = empty_paras
    return model


def get_empty_params_comparedwith_sd(model, model_cfg):
    sd_ckpt = model_cfg.sd_checkpoint
    assert os.path.exists(sd_ckpt), (
        "Error: Stable Diffusion checkpoint NOT found at:%s" % sd_ckpt
    )
    expand_to_3d = model_cfg.expand_to_3d if "expand_to_3d" in model_cfg else False
    adapt_keyname = (
        True
        if not expand_to_3d and model_cfg.params.unet_config.params.temporal_attention
        else False
    )
    is_load_refmodel = (
        True if "ref_model_checkpoint" in model_cfg else False
    )  # added for dpo
    model_pretrained, empty_paras = load_from_pretrainedSD_checkpoint(
        model,
        expand_to_3d=expand_to_3d,
        adapt_keyname=adapt_keyname,
        pretained_ckpt=sd_ckpt,
        is_load_refmodel=is_load_refmodel,
    )
    empty_paras = [n.lstrip("model").lstrip(".") for n in empty_paras]
    return model_pretrained, empty_paras


def get_autoresume_path(logdir):
    # logdir = "/home/rliuay/haoyu/research/DPO-videocrafter/results/dpo-10k-fixdatapair/20240606194637_dpo-10k"
    ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    if os.path.exists(ckpt):
        try:
            tmp = torch.load(ckpt, map_location="cpu")
            e = tmp["epoch"]
            gs = tmp["global_step"]
            mainlogger.info(f"[INFO] Resume from epoch {e}, global step {gs}!")
            del tmp
        except:
            try:
                mainlogger.info("Load last.ckpt failed!")
                ckpts = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(logdir, "checkpoints"))
                        if not os.path.isdir(f)
                    ]
                )
                mainlogger.info(f"all avaible checkpoints: {ckpts}")
                ckpts.remove("last.ckpt")
                if "trainstep_checkpoints" in ckpts:
                    ckpts.remove("trainstep_checkpoints")
                ckpt_path = ckpts[-1]
                ckpt = os.path.join(logdir, "checkpoints", ckpt_path)
                mainlogger.info(f"Select resuming ckpt: {ckpt}")
            except ValueError:
                mainlogger.info("Load last.ckpt failed! and there is no other ckpts")

        resume_checkpt_path = ckpt
        mainlogger.info(f"[INFO] resume from: {ckpt}")
    else:
        resume_checkpt_path = None
        mainlogger.info(
            f"[INFO] no checkpoint found in current workspace: {os.path.join(logdir, 'checkpoints')}"
        )
    return resume_checkpt_path


def set_logger(logfile, name="mainlogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
