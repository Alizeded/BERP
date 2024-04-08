import os
from math import ceil
from typing import Any, Dict, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MFCC, MelSpectrogram
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.components.mixed_speech_dataset import MixedSpeechDataset
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MixedSpeechDataModule(LightningDataModule):
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
        raw_path: str,  # mixed_speech.data
        label_path: str,  # mixed_speech_label.data
        data_dir: str = "data/mixed_speech",
        batch_size: int = 20,
        shuffle: bool = True,
        feat_type: str = "gammatone",
        n_fft: int = 1024,
        n_bins: int = 128,
        hop_length: int = 256,
        max_sample_len: int = 320000,
        sample_rate: int = 16000,
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
        self.clip_length = hop_length

        (  # load data
            self.mixed_speech_train,
            self.mixed_speech_val,
            self.mixed_speech_test,
            self.label_train,
            self.label_val,
            self.label_test,
        ) = self._get_item_and_labels(
            data_dir,
            os.path.join(data_dir, raw_path),
            os.path.join(data_dir, label_path),
        )

        print(
            "train size",
            len(self.mixed_speech_train),
            "val size",
            len(self.mixed_speech_val),
            "eval size",
            len(self.mixed_speech_test),
        )
        # print(self.raw_train[:5])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.feature_extractor = None
        # load feature extractor
        if feat_type == "gammatone":
            self.feature_extractor = Gammatonegram(
                sr=sample_rate,
                n_fft=n_fft,
                n_bins=n_bins,
                hop_length=hop_length,
                power=1.0,
            )
        if feat_type == "mel":
            self.feature_extractor = MelSpectrogram(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_bins,
                hop_length=hop_length,
                power=1.0,
            )
        if feat_type == "mfcc":
            self.feature_extractor = MFCC(
                sr=sample_rate,
                n_mfcc=n_bins,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_bins,
                power=1.0,
            )
        if feat_type == "waveform":
            self.feature_extractor = None

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
            self.data_train = MixedSpeechDataset(
                feat=self.mixed_speech_train,
                label=self.label_train,
                feature_extractor=self.feature_extractor,
            )

            self.data_val = MixedSpeechDataset(
                feat=self.mixed_speech_val,
                label=self.label_val,
                feature_extractor=self.feature_extractor,
            )

            self.data_test = MixedSpeechDataset(
                feat=self.mixed_speech_test,
                label=self.label_test,
                feature_extractor=self.feature_extractor,
            )

            self.dataset = ConcatDataset(
                datasets=[self.data_train, self.data_val, self.data_test]
            )

    def downsample_frame(self, x: torch.Tensor, frame_len: int = 256):
        frame_num = ceil((x.shape[-1] / frame_len))
        frame_aggregated = torch.zeros(frame_num)
        for n in range(frame_num):
            if n != frame_num - 1:
                frame = (
                    x[n * frame_len : (n + 1) * frame_len].float().mean().round().int()
                )
            else:
                frame = x[n * frame_len].int()
            frame_aggregated[n] = frame
        return frame_aggregated.long()

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
        labels = [
            self.downsample_frame(sample["label"], self.clip_length) for sample in batch
        ]
        # for spectrogram-based feature (C, T), for waveform-based feature (T)
        sizes = [i.shape[-1] for i in sources]
        label_sizes = [i.shape[-1] for i in labels]

        # print("sizes", sizes)
        # print("label_sizes", label_sizes)

        if self.feature_extractor is not None:
            target_size = min(max(sizes), ceil(self.max_sample_len / self.clip_length))
            collated_sources = sources[0].new_zeros(
                len(sources), sources[0].shape[0], target_size
            )
        else:
            target_size = min(max(sizes), self.max_sample_len)
            collated_sources = sources[0].new_zeros(len(sources), target_size)

        target_label_size = min(
            max(label_sizes), ceil(self.max_sample_len / self.clip_length)
        )
        # collated_labels = labels[0].new_zeros(len(labels), target_label_size)
        collated_labels = torch.full(
            (len(labels), target_label_size),
            -1,
            dtype=labels[0].dtype,
            device=labels[0].device,
        )

        padding_mask = torch.BoolTensor(
            collated_sources.shape[0], collated_sources.shape[-1]
        ).fill_(False)
        label_padding_mask = torch.BoolTensor(collated_labels.shape).fill_(False)

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

        for i, (label, size) in enumerate(zip(labels, label_sizes)):
            diff = size - target_label_size
            if diff == 0:
                collated_labels[i, ...] = label
            elif diff < 0:
                collated_labels[i, ...] = torch.cat(
                    [label, label.new_full((-diff,), 0.0)], dim=-1
                )
                label_padding_mask[i, diff:] = True
            else:
                collated_labels[i, ...] = self.crop_to_max_size(
                    label, target_label_size, dim=-1
                )

        output = {
            "groundtruth": {
                "target": collated_labels,
                "target_padding_mask": label_padding_mask,
            },
            "net_input": {"source": collated_sources, "padding_mask": padding_mask},
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
        pass

    def _get_item_and_labels(self, data_dir, path_raw, path_label):
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

            # manifest for reverb waveforms
            mixed_speech_train = [
                os.path.join(path_raw, idx)
                for idx in manifest_train_df["mixed_speech"].tolist()
            ]
            mixed_speech_val = [
                os.path.join(path_raw, idx)
                for idx in manifest_val_df["mixed_speech"].tolist()
            ]
            mixed_speech_test = [
                os.path.join(path_raw, idx)
                for idx in manifest_test_df["mixed_speech"].tolist()
            ]

            # manifest for labels
            label_train = [
                os.path.join(path_label, idx)
                for idx in manifest_train_df["mixed_speech_label"].tolist()
            ]
            label_val = [
                os.path.join(path_label, idx)
                for idx in manifest_val_df["mixed_speech_label"].tolist()
            ]
            label_test = [
                os.path.join(path_label, idx)
                for idx in manifest_test_df["mixed_speech_label"].tolist()
            ]

            return (
                mixed_speech_train,
                mixed_speech_val,
                mixed_speech_test,
                label_train,
                label_val,
                label_test,
            )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "ReverbSpeech.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
