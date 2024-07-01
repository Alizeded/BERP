
<div align="center"> <h1>BERP: A Blind Estimator of Room acoustic and physical Parameters for Single-Channel Noisy Speech Signals</h1> </div>

# The official implementation of BERP

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2405.04476-B31B1B.svg)](http://arxiv.org/abs/2405.04476)
<!-- [![Journal](http://img.shields.io/badge/Journal-2024-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

The implementation of Blind Estimator of Room Acoustic and Physical Parameters (BERP) is Pytorch-based framework for predicting room acoustic and physical parameters all-in-one. The project is based on PyTorch Lightning and Hydra. This implementation includes the data preprocessing pipelines, model architectures, training and inference strategies, and experimental configurations.

## Installation

### Pre-requisites

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# add conda to PATH
echo "export PATH=~/miniconda3/bin:$PATH" >> ~/.zshrc
source ~/.zshrc

# initialize conda
conda init zsh

# create conda environment
conda create -n acoustic-toolkit python=3.11.8
conda activate acoustic-toolkit
```

### pdm installation

#### For better dependency management, we use `pdm` as the package manager and deprecate `pip`. You can install `pdm` with the following command

```bash
pip install pdm
```

```bash
# clone project
git clone https://github.com/Alizeded/BERP
cd BERP

# create conda environment and install dependencies
pdm config venv.backend conda # choose the backend as conda
pdm sync # install dependencies with locking dependencies versions
```

## Data download and preprocessing

The data is also avaliable, if it is expired, please contact the authors for more information.

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

Juypiter notebook `data_preprocessing.ipynb` in `notebook` folder details the data preprocessing pipeline.

## How to run

Train model with the default configurations

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
python src/train_jointRegressor.py trainer=ddp data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train_numEstimator.py trainer=ddp logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on quad GPUs
# for unified module
python src/train_jointRegressor.py trainer=ddp trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train.py trainer=ddp trainer.devices=4 logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on multiple GPUs with multiple nodes (2 nodes, 4 GPUs as an example)
# for unified module
python src/train_jointRegressor.py trainer=ddp trainer.nodes=2 trainer.devices=4 data=ReverbSpeechJointEst logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train_numEstimator.py trainer=ddp trainer.nodes=2 trainer.devices=4 logger=wandb_numEstimator callbacks=default_numEstimator
```

## Configuration of training

Please refer to `model`, `callback` and `logger` folder and `train.yaml` in `configs` directory for more details.

## Inference with the trained model

```bash
python src/inference_jointRegressor.py data=ReverbSpeechJointEst
```

```bash
python src/inference_numEstimator.py
```

More details about the inference can be found in `inference.yaml` in `configs` directory.

After inferencing from the trained model, you can use the following command to inference the room acoustic parameters using SSIR model.

```bash
python src/inference_rap_joint.py
```

More details about the inference of room acoustic parameters can be found in `inference_rap.yaml` in `configs` directory.

## Configuration of inference output from the trained model

Please refer to `inference.yaml` in `configs` directory for more details.



## Weights are also available, please check the `weights` directory for more information

In the `weights` directory, you can download the corresponding weights of each module for the BERP framework,
including the unified module and the occupancy module with three featurization methods and the separate module with MFCC featurization.

you can download the weights from the following link:

```bash
# download the weights for the unified module
sh unified_module_Gammatone.sh
sh unified_module_MFCC.sh
sh unified_module_Mel.sh
```

```bash
# download the weights for the occupancy module
sh occupancy_module_Gammatone.sh
sh occupancy_module_MFCC.sh
sh occupancy_module_Mel.sh
```

```bash
# download the weights for the separate module
sh rir_module_MFCC.sh
sh volume_module_MFCC.sh
sh distance_module_MFCC.sh
sh orientation_module_MFCC.sh
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project obtained the great favours from Jianan Chen, our good friend. Thanks for his great help.

## Citation

If you find this repository useful in your research, or if you want to refer to the methodology and code, please cite the following paper:

```bibtex
@misc{wang2024berp,
      title={BERP: A Blind Estimator of Room Acoustic and Physical Parameters for Single-Channel Noisy Speech Signals}, 
      author={Lijun Wang and Yixian Lu and Ziyan Gao and Kai Li and Jianqiang Huang and Yuntao Kong and Shogo Okada},
      year={2024},
      eprint={2405.04476},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
