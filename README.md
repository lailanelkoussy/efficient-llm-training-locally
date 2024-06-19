# efficient-llm-training-locally
This repository contains code to efficiently train huggingface transformers locally transformers, peft, and deepspeed. 
It is almost fully inspired by some examples in the (peft github)[https://github.com/huggingface/peft ]  with many simplification steps in order to make it as compact of an example as possible.
It is a great mine of ressources and I invite everyone interested to dive into it. 

To run this code, it is assumed that it is done in a Linux environment (ubuntu) with at least one NVIDIA GPU available.

## Setting up the environment
### First time setup (environment creation + activation)
```bash
conda create -n training_env_name python=3.11 anaconda
conda activate env_name 
cd [project_directory_root]
pip install -r requirements.txt
```

### Environment activation (not first setup)
```bash
conda activate training_env_name 
cd [project_directory_root]
```

## Training

Steps to follow :
1. Choose which type of training you want to do (QLoRA + ZeRO3 or LoRA + ZeRO3)and modify its .sh file in the corresponding directory as follows :
    - change parameter `model_name_or_path` to model to train
    - change parameter `dataset_name` to training dataset path if it is local or to dataset name if hosted on the huggingface hub
    - change parameter `chat_template_format` if wanting to use one of supported chat_templates, leave `--chat_template_format "custom"` if you want to set your own preprocessing to the dataset
    - change parameter `dataset_text_field` to match the name of the column resulting from preprocessing, or just column name to use for learning if no preprocessing occurred
    - change parameter `output_dir` to directory name for model saving
    - change other parameters as required (epoch, learning rate, LoRA-related parameters, etc.) Note that some changes will require changing corresponding config files (`gradient_accumulation_steps` for example)
2. Change `utils.py` as follows:
    -  Change this line to match required preprocessing (if `chat_template_format` in the .sh script is left as is, note that the output column's name is what should be used for the `dataset_text_field` parameter)
   ```python
   join = lambda x: {"learning_column": "column 1:" + x['column_1'] + "\n\n column 2: " + x["column_2"]}
   ```
3. Change `train.py` as follows:
    - Change these two environment variables at beginning if needed (usually best to change experiment names, but probably no need to change tracking uri)
   ```python 
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "mlflow_experiment_name"
    os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
   ```
4. Adjust the corresponding `deepspeed-config.yaml` as needed. The main parameters that are prone to change are : 
    - `gradient_accumulation_steps` which need to match the parameter in the .sh file
    - `num_processes` which should match the number of GPUs available for training
   
5. At project root, run:
   ```bash
   mlflow ui
   ```
   Mlflow is used to track training parameters. 

6. In separate terminal from project root (change `path/to/sh_script.sh` to the path and name of sh script corresponding to the training you want):
   ```bash
   chmod +x path/to/sh_script.sh
   path/to/sh_script.sh
   ```
