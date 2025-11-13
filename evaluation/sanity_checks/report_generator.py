import os
import glob
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tensorflow as tf

def generate_diagnostics_report(run_dir, output_path=None):
    """
    Combines diagnostic figures and metrics into a PDF summary.
    Automatically finds plots under 'diagnostics/' and logs scalar summaries.
    """
    diag_dir = os.path.join(run_dir, "diagnostics")
    if not os.path.exists(diag_dir):
        print(f"[Report] No diagnostics found in {run_dir}")
        return None

    if output_path is None:
        output_path = os.path.join(run_dir, "diagnostics_summary.pdf")

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Diagnostics Summary Report</b>", styles["Title"]))
    story.append(Paragraph(f"Run directory: {run_dir}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Load scalar summaries if TensorBoard logs exist
    logs_dir = os.path.join(run_dir, "logs")
    if os.path.exists(logs_dir):
        story.append(Paragraph("<b>Scalar Diagnostics (from TensorBoard)</b>", styles["Heading2"]))
        summary = extract_scalar_metrics(logs_dir)
        for key, value in summary.items():
            story.append(Paragraph(f"{key}: {value:.4f}", styles["Normal"]))
        story.append(Spacer(1, 12))

    # Add diagnostic figures
    images = sorted(glob.glob(os.path.join(diag_dir, "*.png")))
    if images:
        story.append(Paragraph("<b>Diagnostic Visualizations</b>", styles["Heading2"]))
        for img_path in images:
            story.append(Image(img_path, width=400, height=300))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No diagnostic images found.", styles["Normal"]))

    doc.build(story)
    print(f"[Report] Saved diagnostic summary: {output_path}")
    return output_path


def extract_scalar_metrics(logs_dir):
    """Extracts last scalar values from TensorBoard logs."""
    summary_iter = tf.compat.v1.train.summary_iterator
    metrics = {}
    for event_file in glob.glob(os.path.join(logs_dir, "**/events.*"), recursive=True):
        for e in summary_iter(event_file):
            for v in e.summary.value:
                metrics[v.tag] = v.simple_value
    return metrics

"""
Usage:
from evaluation.sanity_checks.report_generator import generate_diagnostics_report

generate_diagnostics_report("experiments/runs/supcon_defects/")

Produces: experiments/runs/supcon_defects/diagnostics_summary.pdf
"""