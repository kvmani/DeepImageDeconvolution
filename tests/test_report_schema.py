from pathlib import Path

from reports.summarize_results.scripts.validate_report import validate_report_payload


def test_validate_report_payload_with_status_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "monitoring").mkdir(parents=True)
    (run_dir / "monitoring" / "loss.png").write_text("x")
    (run_dir / "monitoring" / "qual.png").write_text("x")

    report = {
        "run_id": "run_001",
        "timestamp": "2024-01-01T00:00:00",
        "git_commit": "deadbeef",
        "stage": "train",
        "dataset": "synthetic",
        "dataset_path": "data/synthetic",
        "config": "outputs/run_001/config_used.json",
        "metrics": {"val_loss": 0.1},
        "figures": {
            "loss_curve": "monitoring/loss.png",
            "qual_grid": "monitoring/qual.png",
        },
        "status": "running",
        "progress": {"epoch": 1, "epochs_total": 2, "global_step": 10},
        "artifacts": {"history": "history.json", "metrics_csv": "history.csv"},
    }

    errors, _ = validate_report_payload(
        report, "run_001", run_dir, tmp_path, strict_figures=True
    )
    assert errors == []
