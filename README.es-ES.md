

<div align="center"> <h1>BERP: Un Estimador Ciego de Parámetros de Sala para Señales de Voz Ruidosas de Canal Único</h1> </div>

# Implementación oficial de BERP

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2405.04476-B31B1B.svg)](http://arxiv.org/abs/2405.04476)
<!-- [![Journal](http://img.shields.io/badge/Journal-2024-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Descripción

La implementación del Estimador Ciego de Parámetros de Sala (BERP) es un framework basado en PyTorch para predecir, en un solo sistema, los parámetros acústicos y físicos de la sala. El proyecto se basa en PyTorch Lightning y Hydra. Esta implementación incluye las tuberías de preprocesamiento de datos, las arquitecturas del modelo, las estrategias de entrenamiento e inferencia, y las configuraciones experimentales.

## Instalación

### Prerrequisitos

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
conda create -n acoustic-toolkit python=3.11.12
conda activate acoustic-toolkit
```

### Instalación de pdm

#### Para una mejor gestión de dependencias, utilizamos `pdm` como administrador de paquetes y depreciamos `pip`. Puedes instalar `pdm` con el siguiente comando

```bash
pip install pdm
```

```bash
# clone project
git clone https://github.com/Alizeded/BERP
cd BERP

# create conda environment and install dependencies
pdm config venv.backend conda # choose the backend as conda, current we support torch==2.7.0
pdm sync -G cu126 -G toolbox -G logging -G integration # default cuda is 12.6, you can change it to your, e.g. -G cu121
# if you use the older cuda, use this to update pdm.lock
pdm lock -G cu118 -G toolbox -G logging -G integration && pdm sync -G cu118 -G toolbox -G logging -G integration
```

## Descarga y preprocesamiento de datos

Los datos también están disponibles, puedes descargarlos desde el almacenamiento en la nube

```bash
https://jstorage.app.box.com/v/berp-datasets
```

Luego, descomprime los datos y colócalos en el directorio `data`.

El notebook de Jupyter `data_preprocessing.ipynb` y `mix_real_record_preprocess.py` en la carpeta `notebook`, y `synthesize_rir_speech` y `synthesize_speech_noise` en la carpeta `script`, detallan la tubería de preprocesamiento de datos.

## Cómo ejecutar

Entrena el modelo con las configuraciones por defecto en la carpeta `configs`.

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

## Configuración del entrenamiento

Para más detalles, consulta las carpetas `model`, `callback` y `logger`, y el archivo `train.yaml` en el directorio `configs`.

## Inferencia con el modelo entrenado

```bash
# default inference with MFCC featurization
python src/inference_jointRegressor.py
```

```bash
# default inference with MFCC featurization
python src/inference_numEstimator.py
```

Más detalles sobre la inferencia se pueden encontrar en `inference.yaml` dentro del directorio `configs`.

## Los pesos también están disponibles, puedes descargarlos desde los siguientes enlaces:

```bash
# you can copy & paste the following cloud storage link to your browser
# unified module with four types of featurizers
https://jstorage.box.com/s/3164ikshkfml1apsb1diva4h3s7bhmww

# occupancy module with four types of featurizers
https://jstorage.box.com/s/x6ac1z6n982jftb6jsnrqqazxn3iscx
```

Una vez obtenidos los pesos, consulta `eval.yaml` o `inference.yaml` en el directorio `configs` para colocar los pesos en la ruta correcta para la evaluación o inferencia.

PS2: Hemos verificado la validez del enlace de descarga, no debería haber ningún problema. Estamos trabajando en migrar el conjunto de datos al hub de hugginggface dataset hub. Mantente atento.

## Licencia

Este proyecto está licenciado bajo la Licencia GPL-3.0 - consulta el archivo [LICENSE](LICENSE) para más detalles.

## Agradecimientos

Este proyecto ha recibido el gran apoyo de Jianan Chen, nuestro buen amigo. Gracias por su valiosa ayuda.

## Cita

Si consideras útil este repositorio en tu investigación, o si deseas referirte a la metodología y el código, por favor cita el siguiente artículo:

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
