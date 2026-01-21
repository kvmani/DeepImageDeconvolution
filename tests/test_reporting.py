from pathlib import Path

from src.utils.reporting import update_image_log, write_image_log_html, write_metrics_csv


def test_write_image_log_html(tmp_path: Path) -> None:
    entry = {
        "epoch": 1,
        "split": "val",
        "samples": [
            {
                "sample_id": "sample_000001",
                "images": {"C": "epoch_001/sample_000001/C.png"},
                "metrics": {"psnr_a": 10.0},
            }
        ],
    }

    entries = update_image_log(tmp_path, entry)
    write_image_log_html(tmp_path, entries)

    html_path = tmp_path / "index.html"
    assert html_path.exists()
    content = html_path.read_text()
    assert "Epoch 1" in content
    assert "sample_000001" in content


def test_write_metrics_csv(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 0.5},
        {"epoch": 2, "train_loss": 0.25, "val_loss": 0.3},
    ]
    csv_path = tmp_path / "history.csv"
    assert write_metrics_csv(history, csv_path) is True
    content = csv_path.read_text().splitlines()
    assert content[0].startswith("epoch")
    assert "train_loss" in content[0]
