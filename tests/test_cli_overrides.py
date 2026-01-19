from scripts import run_train
from src.utils.config import deep_update


def test_cli_set_overrides_take_precedence() -> None:
    base_config = {
        "train": {"lr": 0.001, "epochs": 10},
        "data": {"root_dir": "data/base"},
    }
    programmatic_overrides = {"train": {"lr": 0.01}}
    config = deep_update(base_config, programmatic_overrides)

    set_overrides = run_train.parse_set_overrides(
        ["train.lr=0.1", "train.epochs=2", "data.root_dir=./datasets/exp1"]
    )
    config = deep_update(config, set_overrides)

    assert config["train"]["lr"] == 0.1
    assert config["train"]["epochs"] == 2
    assert config["data"]["root_dir"] == "./datasets/exp1"
