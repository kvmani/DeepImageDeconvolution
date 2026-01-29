from src.datasets.kikuchi_pairs import split_dataset


class _DummyDataset:
    def __init__(self, sample_ids: list[str]) -> None:
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {"sample_id": self.sample_ids[idx]}


def test_split_dataset_forces_validation_ids() -> None:
    dataset = _DummyDataset([f"id_{idx:02d}" for idx in range(5)])
    train_set, val_set, info = split_dataset(
        dataset, val_split=0.0, seed=123, force_val_ids=["id_03"], return_info=True
    )

    assert val_set is not None
    assert "id_03" in info["val_sample_ids"]
    assert "id_03" not in info["train_sample_ids"]
    assert set(info["train_sample_ids"]).isdisjoint(set(info["val_sample_ids"]))
    assert len(train_set) + len(val_set) == len(dataset)


def test_split_dataset_missing_forced_ids_still_creates_val() -> None:
    dataset = _DummyDataset([f"id_{idx:02d}" for idx in range(4)])
    _, val_set, info = split_dataset(
        dataset, val_split=0.0, seed=5, force_val_ids=["missing"], return_info=True
    )

    assert val_set is not None
    assert len(info["val_sample_ids"]) == 1
    assert info["missing_val_sample_ids"] == ["missing"]


def test_split_dataset_respects_forced_ids_over_val_split() -> None:
    dataset = _DummyDataset([f"id_{idx:02d}" for idx in range(10)])
    _, _, info = split_dataset(
        dataset,
        val_split=0.1,
        seed=7,
        force_val_ids=["id_01", "id_02"],
        return_info=True,
    )

    assert set(["id_01", "id_02"]).issubset(set(info["val_sample_ids"]))
    assert len(info["val_sample_ids"]) >= 2
