import os
from math import e
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


class ReverbSpeechDataModuleBaselineSyn(LightningDataModule):
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
        max_sample_sec: int = 30,  # 30 seconds
        sample_rate: int = 16000,  # 16 kHz
        batch_size: int = 64,
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
        self.max_sample_size = max_sample_sec * sample_rate

        (  # load data
            self.raw_test,
            self.labels_test,
        ) = self._get_item_and_labels()

        print(
            len(self.raw_test),
        )
        # print(self.raw_train[:5])

        self.Th_test = self.labels_test["Th"]

        self.Tt_test = self.labels_test["Tt"]

        self.volume_test = self.labels_test["volume"]

        self.sti_test = self.labels_test["sti"]

        self.alcons_test = self.labels_test["alcons"]

        self.t60_test = self.labels_test["t60"]

        self.edt_test = self.labels_test["edt"]

        self.c80_test = self.labels_test["c80"]

        self.c50_test = self.labels_test["c50"]

        self.d50_test = self.labels_test["d50"]

        self.ts_test = self.labels_test["ts"]

        self.volume_ns_test = self.labels_test["volume_ns"]

        self.dist_src_test = self.labels_test["dist_src"]

        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

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
        if not self.data_test and not self.data_predict:
            self.data_test = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                sti=self.sti_test,
                alcons=self.alcons_test,
                t60=self.t60_test,
                edt=self.edt_test,
                c80=self.c80_test,
                c50=self.c50_test,
                d50=self.d50_test,
                ts=self.ts_test,
                volume_ns=self.volume_ns_test,
                dist_src=self.dist_src_test,
            )

            self.data_predict = ReverbSpeechDatasetBaseline(
                raw_audio=self.raw_test,
                Th=self.Th_test,
                Tt=self.Tt_test,
                volume=self.volume_test,
                sti=self.sti_test,
                alcons=self.alcons_test,
                t60=self.t60_test,
                edt=self.edt_test,
                c80=self.c80_test,
                c50=self.c50_test,
                d50=self.d50_test,
                ts=self.ts_test,
                volume_ns=self.volume_ns_test,
                dist_src=self.dist_src_test,
            )

            self.dataset = ConcatDataset(
                datasets=[
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
        alcons = [sample["alcons"] for sample in batch]
        t60s = [sample["t60"] for sample in batch]
        edts = [sample["edt"] for sample in batch]
        c80s = [sample["c80"] for sample in batch]
        c50s = [sample["c50"] for sample in batch]
        d50s = [sample["d50"] for sample in batch]
        tss = [sample["ts"] for sample in batch]
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
        input["alcons"] = torch.stack(alcons)
        input["t60"] = torch.stack(t60s)
        input["edt"] = torch.stack(edts)
        input["c80"] = torch.stack(c80s)
        input["c50"] = torch.stack(c50s)
        input["d50"] = torch.stack(d50s)
        input["ts"] = torch.stack(tss)
        input["volume_ns"] = torch.stack(volume_nss)
        input["dist_src"] = torch.stack(dist_srcs)

        return input

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
        if os.path.isfile(os.path.join(self._data_dir, "test_manifest_alt.csv")):

            manifest_test_df = pd.read_csv(
                os.path.join(self._data_dir, "test_manifest_alt.csv")
            )

            raw_test = [
                os.path.join(self._path_raw, idx)
                for idx in manifest_test_df["reverbSpeech"].tolist()
            ]

            # labels
            # Th, Tt
            Th_test = manifest_test_df["Th"].tolist()

            Tt_test = manifest_test_df["Tt"].tolist()

            # volume
            volume_test = manifest_test_df["volume_log10"].tolist()

            # sti
            sti_test = manifest_test_df["STI"].tolist()

            # alcons
            alcons_test = manifest_test_df["ALCONS"].tolist()

            # t60
            t60_test = manifest_test_df["T60"].tolist()

            # edt
            edt_test = manifest_test_df["EDT"].tolist()

            # c80
            c80_test = manifest_test_df["C80"].tolist()

            # c50
            c50_test = manifest_test_df["C50"].tolist()

            # d50
            d50_test = manifest_test_df["D50"].tolist()

            # ts
            ts_test = manifest_test_df["TS"].tolist()

            # volume_ns
            volume_ns_test = manifest_test_df["volume"].tolist()

            # dist_src
            dist_src_test = manifest_test_df["distRcv"].tolist()

            # create labels dict
            labels_test = {
                "Th": Th_test,
                "Tt": Tt_test,
                "volume": volume_test,
                "sti": sti_test,
                "alcons": alcons_test,
                "t60": t60_test,
                "edt": edt_test,
                "c80": c80_test,
                "c50": c50_test,
                "d50": d50_test,
                "ts": ts_test,
                "volume_ns": volume_ns_test,
                "dist_src": dist_src_test,
            }

            return (
                raw_test,
                labels_test,
            )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "data" / "ReverbSpeechBaselineSyn.yaml"
    )
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
