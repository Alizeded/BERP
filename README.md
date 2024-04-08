
<div align="center"> <h1>BERP: A Blind Estimator of Room acoustic and physical Parameters</h1> </div>

# The implementation of BERP

<a1 href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a1>
<a2 href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a2>
<a3 href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a4 href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a4><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
<!-- [![Journal](http://img.shields.io/badge/Journal-2024-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

The implementation of Blind Estimator of Room Acoustic and Physical Parameters (BERP) is Pytorch based for training and evaluating a model for room acoustic and physical parameters estimation all-in-one. The project is based on PyTorch Lightning and Hydra. This implementation includes the data preprocessing pipelines, model architectures, training and inference strategies, and experimental configurations.

## Installation

### Pre-requisites

```bash
# wget miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# add conda to PATH
echo "export PATH=~/miniconda3/bin:$PATH" >> ~/.zshrc
source ~/.zshrc

# initialize conda
conda init zsh
```

### pip installation

```bash
# clone project
git clone https://github.com/Alizeded/BERP
cd acoustic

# create conda environment
conda create -n acoustic-toolkit python=3.11.8
conda activate acoustic-toolkit

# install requirements
pip install -r requirements.txt
```

### pdm installation

```bash
# clone project
git clone https://github.com/Alizeded/BERP
cd BERP

# create conda environment and install dependencies
pdm config venv.backend conda # choose the backend as conda
pdm venv create --name acoustic-toolkit 3.11.8 # create pdm virtual environment
eval $(pdm venv activate acoustic-toolkit) # activate pdm virtual environment
pdm install # install dependencies with locking dependencies versions
```

```bash
# clone project
git clone https://github.com/Alizeded/BERP
cd BERP

# create conda environment and install dependencies
conda env create -f environment.yaml -n acoustic-toolkit

# activate conda environment
conda activate acoustic-toolkit
```

# Data download and preprocessing

The data can be downloaded from the following link:

```bash
# download the data
wget https://jstorage.box.com/v/berp-datasets -O noiseReverbSpeech.zip
wget https://jstorage.box.com/v/berp-datasets -O mixed_speech.zip
```

Then, unzip the data and put it in the `data` directory.

```bash
# unzip the data
unzip noiseReverbSpeech.zip -d data
unzip mixed_speech.zip -d data
```

## How to run

Train model with the default configuration

```bash
# train on single GPU
# for unified module
python src/train_jointRegressor.py trainer=gpu data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train_numEstimator.py trainer=gpu logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on dual GPUs
# for unified module
python src/train.py trainer=ddp data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train.py trainer=ddp data=ReverbSpeechJointEst logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on quad GPUs
# for unified module
python src/train.py trainer=ddp trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train.py trainer=ddp trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on multiple GPUs with multiple nodes (2 nodes, 4 GPUs as an example)
# for unified module
python src/train.py trainer=ddp trainer.nodes=2 trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train.py trainer=ddp trainer.nodes=2 trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash

## Configuration of training

Please refer to `model`, `callback` and `logger` folder and `train.yaml` in `configs` directory for more details.

Evaluate with the trained model

```bash
python src/eval_jointRegressor.py data=ReverbSpeechJointEst logger=wandb_jointRegressor
```

```bash
python src/eval_numEstimator.py data=ReverbSpeechJointEst logger=wandb_numEstimator
```

More details about the evaluation can be found in `eval.yaml` in `configs` directory.

After evaluating the trained model, you can use the following command to evaluate the room acoustic parameters using SSIR model.

```bash
python src/eval_rap_joint.py
```

More details about the evaluation of room acoustic parameters can be found in `eval_rap.yaml` in `configs` directory.

## Configuration of inference output from the trained model

Please refer to `inference.yaml` in `configs` directory for more details.

## Inference the room acoustic parameters from the trained model

After obtained the inference output from the trained model, you can use the following command to get the room acoustic parameters using SSIR model.

First configure the `inference_rap.yaml` in `configs` directory, then run the following command

```bash
python src/inference_rap_joint.py
```

## Weights are also available, please check the `weights` directory for more information.

In the `weights` directory, you can download the corresponding weights of each module for the BERP framework,
including the unified module and the occupancy module with three featurization methods and the separate module with MFCC featurization. 

you can download the weights from the following link:

```bash
# download the weights for the unified module
bash unified_module_Gammatone.sh
bash unified_module_MFCC.sh
bash unified_module_Mel.sh
```

```bash
# download the weights for the occupancy module
bash occupancy_module_Gammatone.sh
bash occupancy_module_MFCC.sh
bash occupancy_module_Mel.sh
```

```bash
# download the weights for the separate module
bash rir_module_MFCC.sh
bash volume_module_MFCC.sh
bash distance_module_MFCC.sh
bash orientation_module_MFCC.sh
```

Juypiter notebook `data_preprocessing.ipynb` details the data preprocessing pipeline.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project obtained the great favours from Jianan Chen, our good friend. Thanks for his great help.
