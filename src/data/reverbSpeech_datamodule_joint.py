from math import ceil
import os
from typing import Any, Dict, Optional
import pandas as pd

import torch
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MelSpectrogram, MFCC

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.data.components.reverb_speech_dataset import ReverbSpeechDataset

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ReverbSpeechDataModuleJointEst(LightningDataModule):
    """`LightningDataModule` for the dataset loading and preprocessing.

    reverb noisy and clean dataset preparation

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        path_raw: str,  # raw data path
        data_dir: str = "data/noiseReverbSpeech",
        batch_size: int = 12,
        shuffle: bool = True,
        feat_type: str = "gammatone",
        n_fft: int = 1024,
        n_bins: int = 128,
        hop_length: int = 256,
        sample_rate: int = 16000,
        max_sample_len: int = 320000,
        num_workers: Optional[int] = os.cpu_count() - 1 if os.cpu_count() > 1 else 0,
        pin_memory: bool = False,
    ):
        """Initialize a DataModule."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if self.hparams.num_workers is None:
            self.hparams.num_workers = 0 if os.cpu_count() > 1 else 0

        self.shuffle = shuffle
        self.max_sample_len = max_sample_len
        self.chunk_length = hop_length

        (  # load data
            self.raw_train,
            self.raw_val,
            self.raw_test,
            self.labels_train,
            self.labels_val,
            self.labels_test,
        ) = self._get_item_and_labels(data_dir, os.path.join(data_dir, path_raw))

        print(
            "train size",
            len(self.raw_train),
            "val size",
            len(self.raw_val),
            "eval size",
            len(self.raw_test),
            "predict size",
            len(self.raw_test),
        )

        self.Th_train = self.labels_train["Th"]
        self.Th_val = self.labels_val["Th"]
        self.Th_test = self.labels_test["Th"]

        self.Tt_train = self.labels_train["Tt"]
        self.Tt_val = self.labels_val["Tt"]
        self.Tt_test = self.labels_test["Tt"]

        self.volume_train = self.labels_train["volume"]
        self.volume_val = self.labels_val["volume"]
        self.volume_test = self.labels_test["volume"]

        self.dist_src_train = self.labels_train["dist_src"]
        self.dist_src_val = self.labels_val["dist_src"]
        self.dist_src_test = self.labels_test["dist_src"]

        self.azimuth_src_train = self.labels_train["azimuth_src"]
        self.azimuth_src_val = self.labels_val["azimuth_src"]  #
        self.azimuth_src_test = self.labels_test["azimuth_src"]

        self.elevation_src_train = self.labels_train["elevation_src"]
        self.elevation_src_val = self.labels_val["elevation_src"]
        self.elevation_src_test = self.labels_test["elevation_src"]

        self.azimuth_classif_train = self.labels_train["azimuth_classif"]
        self.azimuth_classif_val = self.labels_val["azimuth_classif"]
        self.azimuth_classif_test = self.labels_test["azimuth_classif"]

        self.elevation_classif_train = self.labels_train["elevation_classif"]
        self.elevation_classif_val = self.labels_val["elevation_classif"]
        self.elevation_classif_test = self.labels_test["elevation_classif"]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # load feature extractor
        self.feature_extractor = None
        if feat_type == "gammatone":
            self.feature_extractor = Gammatonegram(
                sr=sample_rate,
                n_fft=n_fft,
                n_bins=n_bins,
                hop_length=hop_length,
                power=1.0,
            )

        elif feat_type == "mel":
            self.feature_extractor = MelSpectrogram(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_bins,
                hop_length=hop_length,
                power=1.0,
            )

        elif feat_type == "mfcc":
            self.feature_extractor = MFCC(
                sr=sample_rate,
                n_mfcc=n_bins,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_bins,
                power=1.0,
            )

        elif feat_type == "waveform":
            self.feature_extractor = None

        else:
            raise ValueError(f"Feature type {feat_type} not supported")

    @property
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ReverbSpeechDataset(
                feat=self.raw_train,
                Th=self.Th_train,
                Tt=self.Tt_train,
                volume=self.volume_train,
                dist_src=self.dist_src_train,
                azimuth_src=self.azimuth_src_train,
                elevation_src=self.elevation_src_train,
                azimuth_classif=self.azimuth_classif_train,
                elevation_classif=self.elevation_classif_train,
                feature_extractor=self.feature_extractor,
                norm_amplitude=True,
                normalization=True,
            )

            self.data_val = ReverbSpeechDataset(
                feat=self.raw_val,
                Th=self.Th_val,
                Tt=self.Tt_val,
                volume=self.volume_val,
                dist_src=self.dist_src_val,
                azimuth_src=self.azimuth_src_val,
                elevation_src=self.elevation_src_val,
                azimuth_classif=self.azimuth_classif_val,
                elevation_classif=self.elevation_classif_val,
                feature_extractor=self.feature_extractor,
                norm_amplitude=True,
                normalization=True,
            )

            self.data_test = ReverbSpeechDataset(
                feat=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                dist_src=self.dist_src_test,
                azimuth_src=self.azimuth_src_test,
                elevation_src=self.elevation_src_test,
                azimuth_classif=self.azimuth_classif_test,
                elevation_classif=self.elevation_classif_test,
                feature_extractor=self.feature_extractor,
                norm_amplitude=True,
                normalization=True,
            )

            self.data_predict = ReverbSpeechDataset(
                feat=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                dist_src=self.dist_src_test,
                azimuth_src=self.azimuth_src_test,
                elevation_src=self.elevation_src_test,
                azimuth_classif=self.azimuth_classif_test,
                elevation_classif=self.elevation_classif_test,
                feature_extractor=self.feature_extractor,
                norm_amplitude=True,
                normalization=True,
            )

            self.dataset = ConcatDataset(
                datasets=[
                    self.data_train,
                    self.data_val,
                    self.data_test,
                    self.data_predict,
                ]
            )

    def crop_to_max_size(self, t, target_size, dim=0):
        if dim == -1:
            dim = t.dim() - 1

        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = 0
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    def collate_fn(self, batch):
        # pad batch to the same length
        sources = [sample["feat"] for sample in batch]
        Ths = [sample["Th"] for sample in batch]
        Tts = [sample["Tt"] for sample in batch]
        volumes = [sample["volume"] for sample in batch]
        dist_srcs = [sample["dist_src"] for sample in batch]
        azimuth_srcs = [sample["azimuth_src"] for sample in batch]
        elevation_srcs = [sample["elevation_src"] for sample in batch]
        azimuth_classifs = [sample["azimuth_classif"] for sample in batch]
        elevation_classifs = [sample["elevation_classif"] for sample in batch]

        # for spectrogram-based feature (C, T), for waveform-based feature (T)
        sizes = [s.shape[-1] for s in sources]

        if self.feature_extractor is not None:
            target_size = min(max(sizes), ceil(self.max_sample_len / self.chunk_length))
            collated_sources = sources[0].new_zeros(
                len(sources), sources[0].shape[0], target_size
            )
        else:
            target_size = min(max(sizes), self.max_sample_len)
            collated_sources = sources[0].new_zeros(len(sources), target_size)

        padding_mask = torch.BoolTensor(
            collated_sources.shape[0], collated_sources.shape[-1]
        ).fill_(False)

        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i, ...] = source
            elif diff < 0:
                collated_sources[i, ...] = (
                    torch.cat(
                        [source, source.new_full((source.shape[0], -diff), 0.0)],
                        dim=-1,
                    )
                    if source.dim() == 2
                    else torch.cat([source, source.new_full((-diff,), 0.0)], dim=-1)
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i, ...] = self.crop_to_max_size(
                    source, target_size, dim=-1
                )

        output = {
            "groundtruth": {
                "Th": torch.stack(Ths),
                "Tt": torch.stack(Tts),
                "volume": torch.stack(volumes),
                "dist_src": torch.stack(dist_srcs),
                "azimuth": torch.stack(azimuth_srcs),
                "elevation": torch.stack(elevation_srcs),
                "azimuth_classif": torch.stack(azimuth_classifs),
                "elevation_classif": torch.stack(elevation_classifs),
            },
            "net_input": {
                "source": collated_sources,
                "padding_mask": padding_mask,
            },
        }

        return output

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        return super().teardown(stage=stage)

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}  # self.hparams

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        # self.hparams = state_dict
        # self.state.update(state_dict)
        pass

    def _get_item_and_labels(self, data_dir: str, path_raw: str):
        if (
            os.path.isfile(os.path.join(data_dir, "train_manifest.csv"))
            and os.path.isfile(os.path.join(data_dir, "test_manifest.csv"))
            and os.path.isfile(os.path.join(data_dir, "val_manifest.csv"))
        ):
            manifest_train_df = pd.read_csv(
                os.path.join(data_dir, "train_manifest.csv")
            )
            manifest_val_df = pd.read_csv(os.path.join(data_dir, "val_manifest.csv"))
            manifest_test_df = pd.read_csv(os.path.join(data_dir, "test_manifest.csv"))

            # manifest for noisy waveforms
            raw_train = [
                os.path.join(path_raw, idx)
                for idx in manifest_train_df["reverbSpeech"].tolist()
            ]
            raw_val = [
                os.path.join(path_raw, idx)
                for idx in manifest_val_df["reverbSpeech"].tolist()
            ]
            raw_test = [
                os.path.join(path_raw, idx)
                for idx in manifest_test_df["reverbSpeech"].tolist()
            ]

            # labels
            # Th, Tt
            Th_train = manifest_train_df["Th_unitary"].tolist()
            Th_val = manifest_val_df["Th_unitary"].tolist()
            Th_test = manifest_test_df["Th_unitary"].tolist()

            Tt_train = manifest_train_df["Tt"].tolist()
            Tt_val = manifest_val_df["Tt"].tolist()
            Tt_test = manifest_test_df["Tt"].tolist()

            # volume
            volume_train = manifest_train_df["volume_log10_unitary"].tolist()
            volume_val = manifest_val_df["volume_log10_unitary"].tolist()
            volume_test = manifest_test_df["volume_log10_unitary"].tolist()

            # dist_src
            dist_src_train = manifest_train_df["distRcv_norm"].tolist()
            dist_src_val = manifest_val_df["distRcv_norm"].tolist()
            dist_src_test = manifest_test_df["distRcv_norm"].tolist()

            azimuth_src_train = manifest_train_df["azimuth_ori_norm"].tolist()
            elevation_src_train = manifest_train_df["elevation_ori_norm"].tolist()
            azimuth_src_val = manifest_val_df["azimuth_ori_norm"].tolist()
            elevation_src_val = manifest_val_df["elevation_ori_norm"].tolist()
            azimuth_src_test = manifest_test_df["azimuth_ori_norm"].tolist()
            elevation_src_test = manifest_test_df["elevation_ori_norm"].tolist()

            # ori_src binary classification labels
            azimuth_classif_train = manifest_train_df["azimuth_classifier"].tolist()
            elevation_classif_train = manifest_train_df["elevation_classifier"].tolist()
            azimuth_classif_val = manifest_val_df["azimuth_classifier"].tolist()
            elevation_classif_val = manifest_val_df["elevation_classifier"].tolist()
            azimuth_classif_test = manifest_test_df["azimuth_classifier"].tolist()
            elevation_classif_test = manifest_test_df["elevation_classifier"].tolist()

            # create labels dict
            labels_train = {
                "Th": Th_train,
                "Tt": Tt_train,
                "volume": volume_train,
                "dist_src": dist_src_train,
                "azimuth_src": azimuth_src_train,
                "elevation_src": elevation_src_train,
                "azimuth_classif": azimuth_classif_train,
                "elevation_classif": elevation_classif_train,
            }

            labels_val = {
                "Th": Th_val,
                "Tt": Tt_val,
                "volume": volume_val,
                "dist_src": dist_src_val,
                "azimuth_src": azimuth_src_val,
                "elevation_src": elevation_src_val,
                "azimuth_classif": azimuth_classif_val,
                "elevation_classif": elevation_classif_val,
            }

            labels_test = {
                "Th": Th_test,
                "Tt": Tt_test,
                "volume": volume_test,
                "dist_src": dist_src_test,
                "azimuth_src": azimuth_src_test,
                "elevation_src": elevation_src_test,
                "azimuth_classif": azimuth_classif_test,
                "elevation_classif": elevation_classif_test,
            }

            return (
                raw_train,
                raw_val,
                raw_test,
                labels_train,
                labels_val,
                labels_test,
            )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "data" / "ReverbSpeechJointEst.yaml"
    )
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
