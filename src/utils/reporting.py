"""HTML reporting utilities for training monitoring."""
from __future__ import annotations

from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def update_image_log(log_dir: Path, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the image log JSON with a new epoch entry."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "image_log.json"
    entries: List[Dict[str, Any]] = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())

    epoch = entry.get("epoch")
    entries = [existing for existing in entries if existing.get("epoch") != epoch]
    entries.append(entry)
    entries = sorted(entries, key=lambda item: item.get("epoch", 0), reverse=True)

    log_path.write_text(json.dumps(entries, indent=2))
    return entries


def write_image_log_html(
    log_dir: Path,
    entries: List[Dict[str, Any]],
    title: str = "Training Image Log",
    history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write an HTML index for the image log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    html_path = log_dir / "index.html"

    history = history or []
    history_sorted = sorted(history, key=lambda item: item.get("epoch", 0))
    sanitized_history: List[Dict[str, Any]] = []
    for entry in history_sorted:
        clean: Dict[str, Any] = {}
        for key, value in entry.items():
            if isinstance(value, float) and not math.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        sanitized_history.append(clean)
    metric_columns = [
        "epoch",
        "train_loss",
        "train_l_ab",
        "train_l_recon",
        "train_l_x",
        "val_loss",
        "val_l_ab",
        "val_l_recon",
        "val_l_x",
        "psnr_a",
        "psnr_b",
        "ssim_a",
        "ssim_b",
        "l2_a",
        "l2_b",
        "l2_recon",
        "x_mae",
        "x_hat_mean",
        "x_hat_std",
        "x_hat_min",
        "x_hat_max",
    ]
    metric_columns = [
        key
        for key in metric_columns
        if key == "epoch" or any(key in entry for entry in sanitized_history)
    ]

    def _format_metric(key: str, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            if key.startswith(("psnr_", "ssim_")):
                return f"{value:.4f}"
            return f"{value:.6f}"
        return escape(str(value))

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        f"<meta charset=\"utf-8\"><title>{escape(title)}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { margin-bottom: 4px; }",
        "h2 { margin-top: 28px; }",
        "table { border-collapse: collapse; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }",
        "th { background: #f5f5f5; }",
        "img { width: 180px; height: auto; display: block; }",
        ".metrics { font-size: 12px; color: #333; }",
        ".charts { display: flex; flex-wrap: wrap; gap: 18px; }",
        ".chart { flex: 1 1 320px; }",
        ".chart canvas { border: 1px solid #ddd; background: #fff; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        f"<p>Updated: {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>",
    ]

    if sanitized_history:
        latest = sanitized_history[-1]
        lines.append("<h2>Per-Epoch Summary</h2>")
        lines.append(
            "<p>Latest epoch: "
            f"{escape(str(latest.get('epoch', '?')))}"
            "</p>"
        )
        lines.append("<table>")
        lines.append("<tr>")
        for key in metric_columns:
            lines.append(f"<th>{escape(key)}</th>")
        lines.append("</tr>")
        for row in sanitized_history:
            lines.append("<tr>")
            for key in metric_columns:
                lines.append(f"<td>{_format_metric(key, row.get(key))}</td>")
            lines.append("</tr>")
        lines.append("</table>")

        lines.append("<h2>Training Evolution</h2>")
        lines.append("<div class=\"charts\">")
        lines.append(
            "<div class=\"chart\"><h3>Train Loss vs Epoch</h3>"
            "<canvas id=\"chart-train-loss\" width=\"520\" height=\"220\"></canvas></div>"
        )
        if any("val_loss" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>Val Loss vs Epoch</h3>"
                "<canvas id=\"chart-val-loss\" width=\"520\" height=\"220\"></canvas></div>"
            )
        if any("psnr_a" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>PSNR (A/B)</h3>"
                "<canvas id=\"chart-psnr\" width=\"520\" height=\"220\"></canvas></div>"
            )
        if any("ssim_a" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>SSIM (A/B)</h3>"
                "<canvas id=\"chart-ssim\" width=\"520\" height=\"220\"></canvas></div>"
            )
        if any("l2_a" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>L2 (A/B)</h3>"
                "<canvas id=\"chart-l2\" width=\"520\" height=\"220\"></canvas></div>"
            )
        if any("l2_recon" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>L2 Recon</h3>"
                "<canvas id=\"chart-l2-recon\" width=\"520\" height=\"220\"></canvas></div>"
            )
        if any("x_mae" in entry for entry in sanitized_history):
            lines.append(
                "<div class=\"chart\"><h3>x MAE</h3>"
                "<canvas id=\"chart-x-mae\" width=\"520\" height=\"220\"></canvas></div>"
            )
        lines.append("</div>")
        lines.append("<script>")
        lines.append(f"const historyData = {json.dumps(sanitized_history)};")
        lines.append(
            "function renderChart(canvasId, title, series) {\n"
            "  const canvas = document.getElementById(canvasId);\n"
            "  if (!canvas) return;\n"
            "  const ctx = canvas.getContext('2d');\n"
            "  ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
            "  const padding = 40;\n"
            "  const usableW = canvas.width - padding * 2;\n"
            "  const usableH = canvas.height - padding * 2;\n"
            "  const filtered = series.map(s => ({\n"
            "    label: s.label,\n"
            "    color: s.color,\n"
            "    data: historyData.map(d => ({ x: d.epoch, y: d[s.key] }))\n"
            "      .filter(p => typeof p.y === 'number')\n"
            "      .sort((a, b) => a.x - b.x)\n"
            "  })).filter(s => s.data.length > 0);\n"
            "  if (filtered.length === 0) {\n"
            "    ctx.fillStyle = '#666';\n"
            "    ctx.font = '12px Arial';\n"
            "    ctx.fillText('No data available', padding, canvas.height / 2);\n"
            "    return;\n"
            "  }\n"
            "  const xMin = Math.min(...filtered.flatMap(s => s.data.map(p => p.x)));\n"
            "  const xMax = Math.max(...filtered.flatMap(s => s.data.map(p => p.x)));\n"
            "  const yMin = Math.min(...filtered.flatMap(s => s.data.map(p => p.y)));\n"
            "  const yMax = Math.max(...filtered.flatMap(s => s.data.map(p => p.y)));\n"
            "  const xSpan = Math.max(xMax - xMin, 1);\n"
            "  const ySpan = Math.max(yMax - yMin, 1e-9);\n"
            "  function xScale(x) { return padding + ((x - xMin) / xSpan) * usableW; }\n"
            "  function yScale(y) { return canvas.height - padding - ((y - yMin) / ySpan) * usableH; }\n"
            "  ctx.strokeStyle = '#333';\n"
            "  ctx.lineWidth = 1;\n"
            "  ctx.beginPath();\n"
            "  ctx.moveTo(padding, padding);\n"
            "  ctx.lineTo(padding, canvas.height - padding);\n"
            "  ctx.lineTo(canvas.width - padding, canvas.height - padding);\n"
            "  ctx.stroke();\n"
            "  ctx.fillStyle = '#333';\n"
            "  ctx.font = '11px Arial';\n"
            "  ctx.fillText('epoch', canvas.width / 2 - 12, canvas.height - 10);\n"
            "  ctx.save();\n"
            "  ctx.translate(12, canvas.height / 2 + 20);\n"
            "  ctx.rotate(-Math.PI / 2);\n"
            "  ctx.fillText(title, 0, 0);\n"
            "  ctx.restore();\n"
            "  filtered.forEach(s => {\n"
            "    ctx.strokeStyle = s.color;\n"
            "    ctx.lineWidth = 1.5;\n"
            "    ctx.beginPath();\n"
            "    s.data.forEach((p, idx) => {\n"
            "      const x = xScale(p.x);\n"
            "      const y = yScale(p.y);\n"
            "      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);\n"
            "    });\n"
            "    ctx.stroke();\n"
            "  });\n"
            "  ctx.font = '11px Arial';\n"
            "  let legendX = padding;\n"
            "  const legendY = padding - 10;\n"
            "  filtered.forEach(s => {\n"
            "    ctx.fillStyle = s.color;\n"
            "    ctx.fillRect(legendX, legendY, 10, 10);\n"
            "    ctx.fillStyle = '#333';\n"
            "    ctx.fillText(s.label, legendX + 14, legendY + 9);\n"
            "    legendX += 14 + ctx.measureText(s.label).width + 16;\n"
            "  });\n"
            "}\n"
            "renderChart('chart-train-loss', 'loss', [\n"
            "  { key: 'train_loss', label: 'train', color: '#1f77b4' }\n"
            "]);\n"
            "if (document.getElementById('chart-val-loss')) {\n"
            "  renderChart('chart-val-loss', 'loss', [\n"
            "    { key: 'val_loss', label: 'val', color: '#ff7f0e' }\n"
            "  ]);\n"
            "}\n"
            "if (document.getElementById('chart-psnr')) {\n"
            "  renderChart('chart-psnr', 'psnr', [\n"
            "    { key: 'psnr_a', label: 'A', color: '#2ca02c' },\n"
            "    { key: 'psnr_b', label: 'B', color: '#d62728' }\n"
            "  ]);\n"
            "}\n"
            "if (document.getElementById('chart-ssim')) {\n"
            "  renderChart('chart-ssim', 'ssim', [\n"
            "    { key: 'ssim_a', label: 'A', color: '#9467bd' },\n"
            "    { key: 'ssim_b', label: 'B', color: '#8c564b' }\n"
            "  ]);\n"
            "}\n"
            "if (document.getElementById('chart-l2')) {\n"
            "  renderChart('chart-l2', 'l2', [\n"
            "    { key: 'l2_a', label: 'A', color: '#17becf' },\n"
            "    { key: 'l2_b', label: 'B', color: '#7f7f7f' }\n"
            "  ]);\n"
            "}\n"
            "if (document.getElementById('chart-l2-recon')) {\n"
            "  renderChart('chart-l2-recon', 'l2 recon', [\n"
            "    { key: 'l2_recon', label: 'recon', color: '#bcbd22' }\n"
            "  ]);\n"
            "}\n"
            "if (document.getElementById('chart-x-mae')) {\n"
            "  renderChart('chart-x-mae', 'x mae', [\n"
            "    { key: 'x_mae', label: 'x', color: '#ff9896' }\n"
            "  ]);\n"
            "}\n"
        )
        lines.append("</script>")
    else:
        lines.append("<h2>Per-Epoch Summary</h2>")
        lines.append("<p>No history data available yet.</p>")

    image_order = ["C", "A_gt", "B_gt", "A_pred", "B_pred", "C_hat"]

    for entry in entries:
        epoch = entry.get("epoch", "?")
        split = entry.get("split", "val")
        lines.append(f"<h2>Epoch {escape(str(epoch))} (split: {escape(split)})</h2>")
        lines.append("<table>")
        lines.append("<tr>")
        lines.append("<th>Sample</th>")
        for label in image_order:
            lines.append(f"<th>{escape(label)}</th>")
        lines.append("</tr>")

        for sample in entry.get("samples", []):
            sample_id = escape(str(sample.get("sample_id", "unknown")))
            metrics = sample.get("metrics", {})
            metrics_lines = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_lines.append(f"{escape(key)}={value:.4f}")
                else:
                    metrics_lines.append(f"{escape(key)}={escape(str(value))}")
            metric_html = "<br>".join(metrics_lines)

            lines.append("<tr>")
            lines.append(
                f"<td><strong>{sample_id}</strong><div class=\"metrics\">{metric_html}</div></td>"
            )
            images = sample.get("images", {})
            for label in image_order:
                rel_path = images.get(label)
                if rel_path:
                    lines.append(
                        f"<td><img src=\"{escape(rel_path)}\" alt=\"{escape(label)}\"></td>"
                    )
                else:
                    lines.append("<td></td>")
            lines.append("</tr>")

        lines.append("</table>")

    lines.append("</body></html>")
    html_path.write_text("\n".join(lines))
