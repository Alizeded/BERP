
<div align="center"> <h1>BERP: A Blind Estimator of Room Parameters for Single-Channel Noisy Speech Signals</h1> </div>

# The official implementation of BERP

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2405.04476-B31B1B.svg)](http://arxiv.org/abs/2405.04476)
<!-- [![Journal](http://img.shields.io/badge/Journal-2024-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

The implementation of Blind Estimator of Room Parameters (BERP) is Pytorch-based framework for predicting room acoustic and physical parameters all-in-one. The project is based on PyTorch Lightning and Hydra. This implementation includes the data preprocessing pipelines, model architectures, training and inference strategies, and experimental configurations.

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

The data is also avaliable, you can download from the cloud storage

```bash
https://jstorage.app.box.com/v/berp-datasets
```

Then, unzip the data and put it in the `data` directory.

Juypiter notebook `data_preprocessing.ipynb` and `preprocess_real_recordings.ipynb` in `notebook` folder and `synthesize_rir_speech` and `synthesize_speech_noise` in `script` folder detail the data preprocessing pipeline.

## How to run

Train model with the default configurations in `configs` folder.

```bash
# train on single GPU (H100 as an example)
# for unified module
python src/train_jointRegressor.py trainer=gpu logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train_numEstimator.py trainer=gpu trainer.precision=bf16-mixed logger=wandb_numEstimator callbacks=default_numEstimator
```

```bash
# train on one node with multiple GPUs (2 GPUs as an example)
# for unified module
python src/train_jointRegressor.py trainer=ddp logger=wandb_jointRegressor callbacks=default_jointRegressor

# for occupancy module
python src/train_numEstimator.py trainer=ddp trainer.precision=bf16-mixed logger=wandb_numEstimator callbacks=default_numEstimator
```

## Configuration of training

Please refer to `model`, `callback` and `logger` folder and `train.yaml` in `configs` directory for more details.

## Inference with the trained model

```bash
# default inference with MFCC featurization
python src/inference_jointRegressor.py
```

```bash
# default inference with MFCC featurization
python src/inference_numEstimator.py
```

More details about the inference can be found in `inference.yaml` in `configs` directory.

## Weights are also available, you can download the weights from the following link:

```bash
# you can copy & paste the following cloud storage link to your browser
# unified module with four types of featurizers
https://jstorage.box.com/s/3164ikshkfml1apsb1diva4h3s7bhmww

# occupancy module with four types of featurizers
https://jstorage.box.com/s/x6ac1z6n982jftb6jsnrqqazxn3iscx
```

After obtaining the weights, please check the `eval.yaml` or `inference.yaml` in the `configs` directory to put the weights in the correct path for the evaluation or inference.

PS2: We have checked the validity of the download link, there should be no problem with the download link. We are working on migrating the dataset to the hugginggface dataset hub. Please stay tuned.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project obtained the great favours from Jianan Chen, our good friend. Thanks for his great help.

## Citation

If you find this repository useful in your research, or if you want to refer to the methodology and code, please cite the following paper:

```bibtex
@misc{wang2024berp,
      title={BERP: A Blind Estimator of Room Parameters for Single-Channel Noisy Speech Signals}, 
      author={Lijun Wang and Yixian Lu and Ziyan Gao and Kai Li and Jianqiang Huang and Yuntao Kong and Shogo Okada},
      year={2025},
      eprint={2405.04476},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
