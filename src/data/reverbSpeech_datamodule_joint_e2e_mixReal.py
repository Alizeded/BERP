import os
import random
from math import ceil
from typing import Any, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MFCC, MelSpectrogram
from nnAudio.features.stft import STFT
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data.components.reverb_speech_dataset_rap import ReverbSpeechDatasetRap
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ReverbSpeechDataModuleJointEstE2EMixReal(LightningDataModule):
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
        path_real: str,  # real data path
        data_dir: str = "data/noiseReverbSpeech",
        data_dir_real: str = "data/BUT_real_recording_samples_11160samples",
        batch_size: int = 12,
        shuffle: bool = True,  # if using sampler, set to False
        replacement: bool = False,  # for weighted random sampler
        feat_type: str = "gammatone",
        n_fft: int = 1024,
        n_bins: int = 128,
        hop_length: int = 256,
        sample_rate: int = 16000,
        max_sample_len: int = 320000,
        num_real_samples: int = 1004,  # or 5022
        norm_amplitude: bool = True,  # for joint or distSrc estimation, set True
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
        self.num_real_samples = num_real_samples
        self.replacement = replacement
        self.norm_amplitude = norm_amplitude

        (  # load data
            self.raw_train,
            self.raw_val,
            self.raw_test,
            self.labels_train,
            self.labels_val,
            self.labels_test,
        ) = self._get_item_and_labels(
            data_dir,
            os.path.join(data_dir, path_raw),
            data_dir_real,
            os.path.join(data_dir_real, path_real),
        )

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

        # RAP
        self.sti_train = self.labels_train["sti"]
        self.sti_val = self.labels_val["sti"]
        self.sti_test = self.labels_test["sti"]

        self.alcons_train = self.labels_train["alcons"]
        self.alcons_val = self.labels_val["alcons"]
        self.alcons_test = self.labels_test["alcons"]

        self.t60_train = self.labels_train["t60"]
        self.t60_val = self.labels_val["t60"]
        self.t60_test = self.labels_test["t60"]

        self.edt_train = self.labels_train["edt"]
        self.edt_val = self.labels_val["edt"]
        self.edt_test = self.labels_test["edt"]

        self.c80_train = self.labels_train["c80"]
        self.c80_val = self.labels_val["c80"]
        self.c80_test = self.labels_test["c80"]

        self.c50_train = self.labels_train["c50"]
        self.c50_val = self.labels_val["c50"]
        self.c50_test = self.labels_test["c50"]

        self.d50_train = self.labels_train["d50"]
        self.d50_val = self.labels_val["d50"]
        self.d50_test = self.labels_test["d50"]

        self.ts_train = self.labels_train["ts"]
        self.ts_val = self.labels_val["ts"]
        self.ts_test = self.labels_test["ts"]

        # RGP
        self.volume_train = self.labels_train["volume"]
        self.volume_val = self.labels_val["volume"]
        self.volume_test = self.labels_test["volume"]

        self.dist_src_train = self.labels_train["dist_src"]
        self.dist_src_val = self.labels_val["dist_src"]
        self.dist_src_test = self.labels_test["dist_src"]

        self.real_recording_train = self.labels_train["real_recording"]

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

        elif feat_type == "spectrogram":
            self.feature_extractor = STFT(
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                freq_bins=n_bins,
                output_format="Magnitude",
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
            self.data_train = ReverbSpeechDatasetRap(
                feat=self.raw_train,
                sti=self.sti_train,
                alcons=self.alcons_train,
                t60=self.t60_train,
                edt=self.edt_train,
                c80=self.c80_train,
                c50=self.c50_train,
                d50=self.d50_train,
                ts=self.ts_train,
                volume=self.volume_train,
                dist_src=self.dist_src_train,
                feature_extractor=self.feature_extractor,
                norm_amplitude=self.norm_amplitude,
                normalization=True,
            )

            self.data_val = ReverbSpeechDatasetRap(
                feat=self.raw_val,
                sti=self.sti_val,
                alcons=self.alcons_val,
                t60=self.t60_val,
                edt=self.edt_val,
                c80=self.c80_val,
                c50=self.c50_val,
                d50=self.d50_val,
                ts=self.ts_val,
                volume=self.volume_val,
                dist_src=self.dist_src_val,
                feature_extractor=self.feature_extractor,
                norm_amplitude=self.norm_amplitude,
                normalization=True,
            )

            self.data_test = ReverbSpeechDatasetRap(
                feat=self.raw_test,
                sti=self.sti_test,
                alcons=self.alcons_test,
                t60=self.t60_test,
                edt=self.edt_test,
                c80=self.c80_test,
                c50=self.c50_test,
                d50=self.d50_test,
                ts=self.ts_test,
                volume=self.volume_test,
                dist_src=self.dist_src_test,
                feature_extractor=self.feature_extractor,
                norm_amplitude=self.norm_amplitude,
                normalization=True,
            )

            self.data_predict = ReverbSpeechDatasetRap(
                feat=self.raw_test,
                sti=self.sti_test,
                alcons=self.alcons_test,
                t60=self.t60_test,
                edt=self.edt_test,
                c80=self.c80_test,
                c50=self.c50_test,
                d50=self.d50_test,
                ts=self.ts_test,
                volume=self.volume_test,
                dist_src=self.dist_src_test,
                feature_extractor=self.feature_extractor,
                norm_amplitude=self.norm_amplitude,
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
        for _ in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    def collate_fn(self, batch):
        # pad batch to the same length
        sources = [sample["feat"] for sample in batch]
        stis = [sample["sti"] for sample in batch]
        alconss = [sample["alcons"] for sample in batch]
        t60s = [sample["t60"] for sample in batch]
        edts = [sample["edt"] for sample in batch]
        c80s = [sample["c80"] for sample in batch]
        c50s = [sample["c50"] for sample in batch]
        d50s = [sample["d50"] for sample in batch]
        tss = [sample["ts"] for sample in batch]
        volumes = [sample["volume"] for sample in batch]
        dist_srcs = [sample["dist_src"] for sample in batch]

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

        for i, (source, size) in enumerate(zip(sources, sizes, strict=False)):
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
                "sti": torch.stack(stis),
                "alcons": torch.stack(alconss),
                "tr": torch.stack(t60s),
                "edt": torch.stack(edts),
                "c80": torch.stack(c80s),
                "c50": torch.stack(c50s),
                "d50": torch.stack(d50s),
                "ts": torch.stack(tss),
                "volume": torch.stack(volumes),
                "dist_src": torch.stack(dist_srcs),
            },
            "net_input": {
                "source": collated_sources,
                "padding_mask": padding_mask,
            },
        }

        return output

    def make_sample_weight(self):
        labelcount_0 = 43430  # synthetic
        labelcount_1 = self.num_real_samples  # real recording
        labelcount = torch.tensor([labelcount_0, labelcount_1])
        class_weight = 1 / labelcount
        sample_weight = [
            class_weight[real_recording] for real_recording in self.real_recording_train
        ]

        return sample_weight

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        sampler = WeightedRandomSampler(
            weights=self.make_sample_weight(),
            num_samples=len(self.data_train),
            replacement=self.replacement,
        )
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
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

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}  # self.hparams

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        # self.hparams = state_dict
        # self.state.update(state_dict)
        pass

    def _get_item_and_labels(
        self, data_dir: str, path_raw: str, data_dir_real: str, path_real: str
    ):
        if (
            os.path.isfile(os.path.join(data_dir, "train_manifest_alt.csv"))
            and os.path.isfile(
                os.path.join(data_dir_real, "real_audio.metadata/train_manifest.csv")
            )
            and os.path.isfile(os.path.join(data_dir, "test_manifest_alt.csv"))
            and os.path.isfile(
                os.path.join(data_dir_real, "real_audio.metadata/test_manifest.csv")
            )
            and os.path.isfile(os.path.join(data_dir, "val_manifest_alt.csv"))
            and os.path.isfile(
                os.path.join(data_dir_real, "real_audio.metadata/val_manifest.csv")
            )
        ):
            manifest_train_df = pd.read_csv(
                os.path.join(data_dir, "train_manifest_alt.csv")
            )
            manifest_val_df = pd.read_csv(
                os.path.join(data_dir, "val_manifest_alt.csv")
            )
            manifest_test_df = pd.read_csv(
                os.path.join(data_dir, "test_manifest_alt.csv")
            )

            manifest_train_df_real = pd.read_csv(
                os.path.join(data_dir_real, "real_audio.metadata/train_manifest.csv")
            )
            manifest_val_df_real = pd.read_csv(
                os.path.join(data_dir_real, "real_audio.metadata/val_manifest.csv")
            )
            manifest_test_df_real = pd.read_csv(
                os.path.join(data_dir_real, "real_audio.metadata/test_manifest.csv")
            )

            # manifest for noisy waveforms
            raw_train_syn = [
                os.path.join(path_raw, idx)
                for idx in manifest_train_df["reverbSpeech"].tolist()
            ]
            raw_val_syn = [
                os.path.join(path_raw, idx)
                for idx in manifest_val_df["reverbSpeech"].tolist()
            ]
            raw_test_syn = [
                os.path.join(path_raw, idx)
                for idx in manifest_test_df["reverbSpeech"].tolist()
            ]

            raw_train_real = [
                os.path.join(path_real, idx)
                for idx in manifest_train_df_real["realRecording"].tolist()
            ]
            raw_val_real = [
                os.path.join(path_real, idx)
                for idx in manifest_val_df_real["realRecording"].tolist()
            ]
            raw_test_real = [
                os.path.join(path_real, idx)
                for idx in manifest_test_df_real["realRecording"].tolist()
            ]

            # combine the two datasets
            raw_train = raw_train_syn + raw_train_real
            raw_val = raw_val_syn + raw_val_real
            raw_test = raw_test_syn + raw_test_real

            # labels
            # RAP
            # STI
            sti_train_syn = manifest_train_df["STI"].tolist()
            sti_val_syn = manifest_val_df["STI"].tolist()
            sti_test_syn = manifest_test_df["STI"].tolist()

            sti_train_real = manifest_train_df_real["STI"].tolist()
            sti_val_real = manifest_val_df_real["STI"].tolist()
            sti_test_real = manifest_test_df_real["STI"].tolist()

            # ALCONS
            alcons_train_syn = manifest_train_df["ALCONS"].tolist()
            alcons_val_syn = manifest_val_df["ALCONS"].tolist()
            alcons_test_syn = manifest_test_df["ALCONS"].tolist()

            alcons_train_real = manifest_train_df_real["ALCONS"].tolist()
            alcons_val_real = manifest_val_df_real["ALCONS"].tolist()
            alcons_test_real = manifest_test_df_real["ALCONS"].tolist()

            # T60
            t60_train_syn = manifest_train_df["T60"].tolist()
            t60_val_syn = manifest_val_df["T60"].tolist()
            t60_test_syn = manifest_test_df["T60"].tolist()

            t60_train_real = manifest_train_df_real["T60"].tolist()
            t60_val_real = manifest_val_df_real["T60"].tolist()
            t60_test_real = manifest_test_df_real["T60"].tolist()

            # EDT
            edt_train_syn = manifest_train_df["EDT"].tolist()
            edt_val_syn = manifest_val_df["EDT"].tolist()
            edt_test_syn = manifest_test_df["EDT"].tolist()

            edt_train_real = manifest_train_df_real["EDT"].tolist()
            edt_val_real = manifest_val_df_real["EDT"].tolist()
            edt_test_real = manifest_test_df_real["EDT"].tolist()

            # C80
            c80_train_syn = manifest_train_df["C80"].tolist()
            c80_val_syn = manifest_val_df["C80"].tolist()
            c80_test_syn = manifest_test_df["C80"].tolist()

            c80_train_real = manifest_train_df_real["C80"].tolist()
            c80_val_real = manifest_val_df_real["C80"].tolist()
            c80_test_real = manifest_test_df_real["C80"].tolist()

            # C50
            c50_train_syn = manifest_train_df["C50"].tolist()
            c50_val_syn = manifest_val_df["C50"].tolist()
            c50_test_syn = manifest_test_df["C50"].tolist()

            c50_train_real = manifest_train_df_real["C50"].tolist()
            c50_val_real = manifest_val_df_real["C50"].tolist()
            c50_test_real = manifest_test_df_real["C50"].tolist()

            # D50
            d50_train_syn = manifest_train_df["D50"].tolist()
            d50_val_syn = manifest_val_df["D50"].tolist()
            d50_test_syn = manifest_test_df["D50"].tolist()

            d50_train_real = manifest_train_df_real["D50"].tolist()
            d50_val_real = manifest_val_df_real["D50"].tolist()
            d50_test_real = manifest_test_df_real["D50"].tolist()

            # TS
            ts_train_syn = manifest_train_df["TS"].tolist()
            ts_val_syn = manifest_val_df["TS"].tolist()
            ts_test_syn = manifest_test_df["TS"].tolist()

            ts_train_real = manifest_train_df_real["TS"].tolist()
            ts_val_real = manifest_val_df_real["TS"].tolist()
            ts_test_real = manifest_test_df_real["TS"].tolist()

            # volume
            volume_train_syn = manifest_train_df["volume_log10"].tolist()
            volume_val_syn = manifest_val_df["volume_log10"].tolist()
            volume_test_syn = manifest_test_df["volume_log10"].tolist()

            volume_train_real = manifest_train_df_real["volume_log10"].tolist()
            volume_val_real = manifest_val_df_real["volume_log10"].tolist()
            volume_test_real = manifest_test_df_real["volume_log10"].tolist()

            # dist_src
            dist_src_train_syn = manifest_train_df["distRcv"].tolist()
            dist_src_val_syn = manifest_val_df["distRcv"].tolist()
            dist_src_test_syn = manifest_test_df["distRcv"].tolist()

            dist_src_train_real = manifest_train_df_real["distRcv"].tolist()
            dist_src_val_real = manifest_val_df_real["distRcv"].tolist()
            dist_src_test_real = manifest_test_df_real["distRcv"].tolist()

            real_recording_syn = [0] * len(raw_train_syn)
            real_recording_real = [1] * len(raw_train_real)

            # concatenate the two datasets
            sti_train = sti_train_syn + sti_train_real
            sti_val = sti_val_syn + sti_val_real
            sti_test = sti_test_syn + sti_test_real

            alcons_train = alcons_train_syn + alcons_train_real
            alcons_val = alcons_val_syn + alcons_val_real
            alcons_test = alcons_test_syn + alcons_test_real

            t60_train = t60_train_syn + t60_train_real
            t60_val = t60_val_syn + t60_val_real
            t60_test = t60_test_syn + t60_test_real

            edt_train = edt_train_syn + edt_train_real
            edt_val = edt_val_syn + edt_val_real
            edt_test = edt_test_syn + edt_test_real

            c80_train = c80_train_syn + c80_train_real
            c80_val = c80_val_syn + c80_val_real
            c80_test = c80_test_syn + c80_test_real

            c50_train = c50_train_syn + c50_train_real
            c50_val = c50_val_syn + c50_val_real
            c50_test = c50_test_syn + c50_test_real

            d50_train = d50_train_syn + d50_train_real
            d50_val = d50_val_syn + d50_val_real
            d50_test = d50_test_syn + d50_test_real

            ts_train = ts_train_syn + ts_train_real
            ts_val = ts_val_syn + ts_val_real
            ts_test = ts_test_syn + ts_test_real

            volume_train = volume_train_syn + volume_train_real
            volume_val = volume_val_syn + volume_val_real
            volume_test = volume_test_syn + volume_test_real

            dist_src_train = dist_src_train_syn + dist_src_train_real
            dist_src_val = dist_src_val_syn + dist_src_val_real
            dist_src_test = dist_src_test_syn + dist_src_test_real

            real_recording = real_recording_syn + real_recording_real

            # create labels dict
            labels_train = {
                "sti": sti_train,
                "alcons": alcons_train,
                "t60": t60_train,
                "edt": edt_train,
                "c80": c80_train,
                "c50": c50_train,
                "d50": d50_train,
                "ts": ts_train,
                "volume": volume_train,
                "dist_src": dist_src_train,
                "real_recording": real_recording,
            }

            labels_val = {
                "sti": sti_val,
                "alcons": alcons_val,
                "t60": t60_val,
                "edt": edt_val,
                "c80": c80_val,
                "c50": c50_val,
                "d50": d50_val,
                "ts": ts_val,
                "volume": volume_val,
                "dist_src": dist_src_val,
            }

            labels_test = {
                "sti": sti_test,
                "alcons": alcons_test,
                "t60": t60_test,
                "edt": edt_test,
                "c80": c80_test,
                "c50": c50_test,
                "d50": d50_test,
                "ts": ts_test,
                "volume": volume_test,
                "dist_src": dist_src_test,
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
