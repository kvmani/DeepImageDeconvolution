from pathlib import Path

from scripts.compare_runs import build_comparison_rows, write_comparison_table


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload))


def test_compare_runs_writes_csv(tmp_path: Path) -> None:
    report_a = {
        "run_id": "run_a",
        "timestamp": "2024-01-01T00:00:00",
        "git_commit": "deadbeef",
        "stage": "train",
        "dataset": "synthetic",
        "dataset_path": "data/synthetic",
        "config": "outputs/run_a/config_used.json",
        "metrics": {"val_loss": 0.2},
        "figures": {"loss_curve": "loss.png", "qual_grid": "qual.png"},
        "status": "complete",
        "progress": {"epoch": 2, "epochs_total": 2, "global_step": 20},
        "artifacts": {"history": "history.json", "metrics_csv": "history.csv"},
    }
    report_b = {
        "run_id": "run_b",
        "timestamp": "2024-01-02T00:00:00",
        "git_commit": "deadbeef",
        "stage": "train",
        "dataset": "synthetic",
        "dataset_path": "data/synthetic",
        "config": "outputs/run_b/config_used.json",
        "metrics": {"val_loss": 0.15},
        "figures": {"loss_curve": "loss.png", "qual_grid": "qual.png"},
        "status": "complete",
        "progress": {"epoch": 2, "epochs_total": 2, "global_step": 20},
        "artifacts": {"history": "history.json", "metrics_csv": "history.csv"},
    }

    report_paths = [
        tmp_path / "outputs" / "run_a" / "report.json",
        tmp_path / "outputs" / "run_b" / "report.json",
    ]
    _write_report(report_paths[0], report_a)
    _write_report(report_paths[1], report_b)

    rows, metrics = build_comparison_rows(report_paths, metrics_keys=["val_loss"])
    out_csv = tmp_path / "run_comparison.csv"
    write_comparison_table(rows, metrics, out_csv, output_format="csv")

    content = out_csv.read_text()
    assert "run_id" in content
    assert "val_loss" in content
