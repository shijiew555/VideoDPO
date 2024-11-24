
# Gallery
<table class="center">
  
  <tr>
    <td style="text-align:center;" width="320">Before Alignment</td>
    <td style="text-align:center;" width="320">After Alignment</td>
  </tr>
  <tr>
    <td><a href="./assets/vc2-init/0105.gif"><img src="./assets/vc2-init/0105.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0105.gif"><img src="./assets/vc2-dpo/0105.gif" width="320"></a></td>
  </tr>
  
  <tr>
    <td style="text-align:center;" width="320">Before Alignment</td>
    <td style="text-align:center;" width="320">After Alignment</td>
  </tr>
  <tr>
    <td><a href="./assets/vc2-init/0163.gif"><img src="./assets/vc2-init/0163.gif" width="320"></a></td>
    <td><a href="./assets/vc2-dpo/0163.gif"><img src="./assets/vc2-dpo/0163.gif" width="320"></a></td>
  </tr>

</table>


# Get Started 

## prepare environments 
```shell
conda create -n videodpo python=3.10 -y
conda activate videodpo
pip install -r requirements.txt
```

## prepare checkpoints
### VideoCrafter2
run following instruction to create initial checkpoints. 

```shell
mkdir -p checkpoints/vc2
wget -P checkpoints/vc2 https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
python utils/create_ref_model.py
```
### T2V-Turbo(V1)

T2V-Turbo is latent consistency model. We provide finetuning LCM based on VC2. Please download vc2 checkpoints first. And then run: 
```shell
mkdir -p checkpoints/t2v-turbo
wget -O checkpoints/t2v-turbo/unet_lora.pt "https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/resolve/main/unet_lora.pt?download=true"
```

## Prepare Training Data 
download vidpro-vc2-dataset.tar from the following link. 
then ln -s the dataset to /data/vidpro-dpo-dataset.
or u could also add dataset with same structure in configs/dpo/vidpro/train_data.yaml

> to reduce peak memory use in training stage, we recommend to disable validation by not providing val_data.yaml.


## Finetune VideoCrafter2
```shell
bash configs/dpo/run.sh
```

## Inference VideoCrafter2
We support inference with different types of inputs and outputs.
We support both json and text formats to read prompts. 

```shell
bash script_sh/inference_t2v.sh
```
## Finetune T2V-Turbo(V1)
```shell
bash configs/dpo/run.sh
```

## Inference T2V-Turbo(V1)

# Helper Functions
besides, we also provide some useful tools to improve your finetuning experiences. 
We could automatically remove training logs without any checkpoints saved. 
```bash 
python utils/clean_results.py -d ./results 
```

# Citation
```
To be updated... 
```