import ast
import os
from typing import Any, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.components.reverb_speech_dataset_baseline import (
    ReverbSpeechDatasetBaseline,
)
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ReverbSpeechDataModuleBaseline(LightningDataModule):
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
        path_clean: str,  # clean data path
        data_dir: str = "data/noiseReverbSpeech",
        max_sample_sec: int = 30,  # 30 seconds
        sample_rate: int = 16000,  # 16 kHz
        batch_size: int = 64,
        shuffle: bool = True,
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

        self._data_dir = data_dir
        self._path_raw = os.path.join(data_dir, path_raw)
        self._path_clean = os.path.join(data_dir, path_clean)
        self.shuffle = shuffle
        self.max_sample_size = max_sample_sec * sample_rate

        (  # load data
            self.raw_train,
            self.raw_val,
            self.raw_test,
            self.labels_train,
            self.labels_val,
            self.labels_test,
        ) = self._get_item_and_labels()

        print(
            "train size",
            len(self.raw_train),
            "val size",
            len(self.raw_val),
            "eval size",
            len(self.raw_test),
        )
        # print(self.raw_train[:5])

        self.Th_train = self.labels_train["Th"]
        self.Th_val = self.labels_val["Th"]
        self.Th_test = self.labels_test["Th"]

        self.Tt_train = self.labels_train["Tt"]
        self.Tt_val = self.labels_val["Tt"]
        self.Tt_test = self.labels_test["Tt"]

        self.volume_train = self.labels_train["volume"]
        self.volume_val = self.labels_val["volume"]
        self.volume_test = self.labels_test["volume"]

        self.sti_train = self.labels_train["sti"]
        self.sti_val = self.labels_val["sti"]
        self.sti_test = self.labels_test["sti"]

        self.t60_train = self.labels_train["t60"]
        self.t60_val = self.labels_val["t60"]
        self.t60_test = self.labels_test["t60"]

        self.c80_train = self.labels_train["c80"]
        self.c80_val = self.labels_val["c80"]
        self.c80_test = self.labels_test["c80"]

        self.c50_train = self.labels_train["c50"]
        self.c50_val = self.labels_val["c50"]
        self.c50_test = self.labels_test["c50"]

        self.volume_ns_train = self.labels_train["volume_ns"]
        self.volume_ns_val = self.labels_val["volume_ns"]
        self.volume_ns_test = self.labels_test["volume_ns"]

        self.dist_src_train = self.labels_train["dist_src"]
        self.dist_src_val = self.labels_val["dist_src"]
        self.dist_src_test = self.labels_test["dist_src"]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

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
            self.data_train = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_train,
                Th=self.Th_train,
                Tt=self.Tt_train,
                volume=self.volume_train,
                sti=self.sti_train,
                t60=self.t60_train,
                c80=self.c80_train,
                c50=self.c50_train,
                volume_ns=self.volume_ns_train,
                dist_src=self.dist_src_train,
            )

            self.data_val = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_val,
                Th=self.Th_val,
                Tt=self.Tt_val,
                volume=self.volume_val,
                sti=self.sti_val,
                t60=self.t60_val,
                c80=self.c80_val,
                c50=self.c50_val,
                volume_ns=self.volume_ns_val,
                dist_src=self.dist_src_val,
            )

            self.data_test = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                sti=self.sti_test,
                t60=self.t60_test,
                c80=self.c80_test,
                c50=self.c50_test,
                volume_ns=self.volume_ns_test,
                dist_src=self.dist_src_test,
            )

            self.data_predict = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                sti=self.sti_test,
                t60=self.t60_test,
                c80=self.c80_test,
                c50=self.c50_test,
                volume_ns=self.volume_ns_test,
                dist_src=self.dist_src_test,
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
        raws = [sample["raw"] for sample in batch]
        Ths = [sample["Th"] for sample in batch]
        Tts = [sample["Tt"] for sample in batch]
        volumes = [sample["volume"] for sample in batch]
        stis = [sample["sti"] for sample in batch]
        t60s = [sample["t60"] for sample in batch]
        c80s = [sample["c80"] for sample in batch]
        c50s = [sample["c50"] for sample in batch]
        volume_nss = [sample["volume_ns"] for sample in batch]
        dist_srcs = [sample["dist_src"] for sample in batch]

        sizes = [len(s) for s in raws]
        target_size = min(max(sizes), self.max_sample_size)

        collated_raws = raws[0].new_zeros((len(raws), target_size))

        for i, (raw, size) in enumerate(zip(raws, sizes, strict=False)):
            diff = size - target_size
            if diff == 0:
                collated_raws[i] = raw
            elif diff < 0:
                collated_raws[i] = torch.cat([raw, raw.new_full((-diff,), 0.0)])
            else:
                collated_raws[i] = self.crop_to_max_size(raw, target_size)

        input = {"raw": collated_raws}
        input["Th"] = torch.stack(Ths)
        input["Tt"] = torch.stack(Tts)
        input["volume"] = torch.stack(volumes)
        input["sti"] = torch.stack(stis)
        input["t60"] = torch.stack(t60s)
        input["c80"] = torch.stack(c80s)
        input["c50"] = torch.stack(c50s)
        input["volume_ns"] = torch.stack(volume_nss)
        input["dist_src"] = torch.stack(dist_srcs)

        return input

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

    def _get_item_and_labels(self):
        if (
            os.path.isfile(os.path.join(self._data_dir, "train_manifest_alt.csv"))
            and os.path.isfile(os.path.join(self._data_dir, "test_manifest_alt.csv"))
            and os.path.isfile(os.path.join(self._data_dir, "val_manifest_alt.csv"))
        ):
            manifest_train_df = pd.read_csv(
                os.path.join(self._data_dir, "train_manifest_alt.csv")
            )
            manifest_val_df = pd.read_csv(
                os.path.join(self._data_dir, "val_manifest_alt.csv")
            )
            manifest_test_df = pd.read_csv(
                os.path.join(self._data_dir, "test_manifest_alt.csv")
            )

            # manifest for noisy waveforms
            raw_train = [
                os.path.join(self._path_raw, idx)
                for idx in manifest_train_df["reverbSpeech"].tolist()
            ]
            raw_val = [
                os.path.join(self._path_raw, idx)
                for idx in manifest_val_df["reverbSpeech"].tolist()
            ]
            raw_test = [
                os.path.join(self._path_raw, idx)
                for idx in manifest_test_df["reverbSpeech"].tolist()
            ]

            # labels
            # Th, Tt
            Th_train = manifest_train_df["Th"].tolist()
            Th_val = manifest_val_df["Th"].tolist()
            Th_test = manifest_test_df["Th"].tolist()

            Tt_train = manifest_train_df["Tt"].tolist()
            Tt_val = manifest_val_df["Tt"].tolist()
            Tt_test = manifest_test_df["Tt"].tolist()

            # volume
            volume_train = manifest_train_df["volume_log10"].tolist()
            volume_val = manifest_val_df["volume_log10"].tolist()
            volume_test = manifest_test_df["volume_log10"].tolist()

            # sti
            sti_train = manifest_train_df["STI"].tolist()
            sti_val = manifest_val_df["STI"].tolist()
            sti_test = manifest_test_df["STI"].tolist()

            # t60
            t60_train = manifest_train_df["T60"].tolist()
            t60_val = manifest_val_df["T60"].tolist()
            t60_test = manifest_test_df["T60"].tolist()

            # c80
            c80_train = manifest_train_df["C80"].tolist()
            c80_val = manifest_val_df["C80"].tolist()
            c80_test = manifest_test_df["C80"].tolist()

            # c50
            c50_train = manifest_train_df["C50"].tolist()
            c50_val = manifest_val_df["C50"].tolist()
            c50_test = manifest_test_df["C50"].tolist()

            # volume_ns
            volume_ns_train = manifest_train_df["volume"].tolist()
            volume_ns_val = manifest_val_df["volume"].tolist()
            volume_ns_test = manifest_test_df["volume"].tolist()

            # dist_src
            dist_src_train = manifest_train_df["distRcv"].tolist()
            dist_src_val = manifest_val_df["distRcv"].tolist()
            dist_src_test = manifest_test_df["distRcv"].tolist()

            # create labels dict
            labels_train = {
                "Th": Th_train,
                "Tt": Tt_train,
                "volume": volume_train,
                "sti": sti_train,
                "t60": t60_train,
                "c80": c80_train,
                "c50": c50_train,
                "volume_ns": volume_ns_train,
                "dist_src": dist_src_train,
            }

            labels_val = {
                "Th": Th_val,
                "Tt": Tt_val,
                "volume": volume_val,
                "sti": sti_val,
                "t60": t60_val,
                "c80": c80_val,
                "c50": c50_val,
                "volume_ns": volume_ns_val,
                "dist_src": dist_src_val,
            }

            labels_test = {
                "Th": Th_test,
                "Tt": Tt_test,
                "volume": volume_test,
                "sti": sti_test,
                "t60": t60_test,
                "c80": c80_test,
                "c50": c50_test,
                "volume_ns": volume_ns_test,
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
        root / "configs" / "data" / "ReverbSpeechBaseline.yaml"
    )
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
