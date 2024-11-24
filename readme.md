
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


# Install
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

## Finetune videocrafter

```shell
bash configs/dpo/run.sh
```

# Inference 
We support inference with different types of inputs and outputs.
We support both json and text formats to read prompts. 

```shell
bash script_sh/inference_t2v.sh
```

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