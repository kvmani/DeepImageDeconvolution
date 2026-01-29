import logging
from pathlib import Path

from src.training.train import ImageLogConfig, ImageLogState, _select_log_indices


class _DummyDataset:
    def __init__(self, sample_ids: list[str]) -> None:
        self._sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {"sample_id": self._sample_ids[idx]}


def _make_cfg(tmp_path: Path, **overrides) -> ImageLogConfig:
    cfg = ImageLogConfig(
        enabled=True,
        interval=1,
        max_samples=2,
        sample_strategy="fixed",
        sample_ids=[],
        split="train",
        output_dir=tmp_path,
        image_format="png",
        write_html=True,
        include_recon=True,
        mask_metrics=False,
        seed=123,
    )
    return ImageLogConfig(**{**cfg.__dict__, **overrides})


def test_select_log_indices_falls_back_when_sample_ids_missing(tmp_path: Path) -> None:
    dataset = _DummyDataset([f"id_{idx:03d}" for idx in range(10)])
    logger = logging.getLogger("test_image_log_selection")

    cfg = _make_cfg(
        tmp_path,
        sample_strategy="first",
        sample_ids=["sample_does_not_exist"],
    )
    state = ImageLogState()

    indices = _select_log_indices(dataset, cfg, state, logger)
    assert indices == [0]

    indices_second = _select_log_indices(dataset, cfg, state, logger)
    assert indices_second == indices


def test_select_log_indices_respects_sample_ids(tmp_path: Path) -> None:
    dataset = _DummyDataset(["alpha", "bravo", "charlie"])
    logger = logging.getLogger("test_image_log_selection")

    cfg = _make_cfg(tmp_path, sample_strategy="fixed", sample_ids=["bravo"])
    state = ImageLogState()

    indices = _select_log_indices(dataset, cfg, state, logger)
    assert indices == [1]
