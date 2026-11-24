"""Build a local progress PDF from the completed phase-seven evidence only.

No data acquisition, inference, training, cloud API, or publication is performed.
Run from the project root with .venv/Scripts/python.exe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "phase7_depth_format"
RESULTS = HOME / "results/nyu_frame_v1"
PDF = ROOT / "output/pdf/Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf"
MANIFEST = HOME / "report_manifest.json"
INK = colors.HexColor("#17324d")
TEAL = colors.HexColor("#0f766e")
MUTED = colors.HexColor("#526476")
LIGHT = colors.HexColor("#eef4f7")
WIDTH = A4[0] - 36 * mm


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    audit = read(RESULTS / "audit.json")
    summary = read(RESULTS / "summary.json")
    pairs = read(RESULTS / "paired.json")
    assert audit["status"] == "PASS"
    assert audit["new_trained_models"] == 9
    assert audit["all_final_checkpoints_reinferred"] == 15
    assert audit["saved_predictions_checked"] == 3240
    assert audit["protocol_sha256"] == sha(HOME / "PROTOCOL.md")
    assert audit["analyzer_sha256"] == sha(HOME / "analyze.py")
    final = {
        (r["condition"], r["group"]): r
        for r in summary if r["selection"] == "fixed_final"
    }
    order = ["native", "camera_real", "camera_unitdepth", "native_flat", "camera_flat"]
    names = {
        "native": "Native XYZ",
        "camera_real": "Canonical camera axes",
        "camera_unitdepth": "Canonical axes + median depth = 1",
        "native_flat": "Native flat control",
        "camera_flat": "Canonical flat control",
    }
    assert all(final[(c, "all")]["n"] == 324 for c in order)
    assert len(pairs) == 5 and sum(p["role"] == "primary" for p in pairs) == 2

    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Report", str(font_dir / "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("ReportBold", str(font_dir / "calibrib.ttf")))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleReport", fontName="ReportBold", fontSize=25,
                              leading=28, textColor=INK, spaceAfter=9))
    styles.add(ParagraphStyle("Kicker", fontName="ReportBold", fontSize=10,
                              leading=13, textColor=TEAL, spaceAfter=6))
    styles.add(ParagraphStyle("Subhead", fontName="ReportBold", fontSize=14,
                              leading=17, textColor=INK, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle("BodyReport", fontName="Report", fontSize=10.5,
                              leading=14, textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle("SmallReport", fontName="Report", fontSize=9,
                              leading=11.5, textColor=MUTED, spaceAfter=6))
    styles.add(ParagraphStyle("CellReport", fontName="Report", fontSize=9,
                              leading=11.4, textColor=INK))
    styles.add(ParagraphStyle("HeaderReport", fontName="ReportBold", fontSize=9,
                              leading=11.4, textColor=colors.white))

    def p(value, style="BodyReport"):
        return Paragraph(value, styles[style])

    def table(headers, rows, widths):
        data = [[p(str(v), "HeaderReport") for v in headers]]
        data.extend([[p(str(v), "CellReport") for v in row] for row in rows])
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#cad7e0")),
        ]))
        return result

    def figure(filename, height):
        return Image(str(RESULTS / "figures" / filename), width=WIDTH, height=height)

    flow = [
        p("RESEARCH PROGRESS / 16 NOVEMBER 2026", "Kicker"),
        p("Does standardizing depth<br/>help PCCC?", "TitleReport"),
        p("Shirsha Mitra | Student ID 2023521460130 | Sichuan University", "SmallReport"),
        p("<b>Completed:</b> nine new models at 10,000 epochs across the official three "
          "NYU-v2 + DIODE folds. The earlier six native/flat reference models were reused. "
          "Evaluation covers 324 captures, including 258 NYU and 66 DIODE captures."),
        p("<b>Main finding:</b> neither primary comparison establishes a benefit from "
          "median-depth normalization. The small axes-only mean improvement is also inconclusive."),
        p("Fixed-final checkpoint results", "Subhead"),
        table(["Input representation", "Mean error", "Median", "Worst 25%"], [
            [names[c], f"{final[(c, 'all')]['mean']:.4f}",
             f"{final[(c, 'all')]['median']:.4f}", f"{final[(c, 'all')]['worst25']:.4f}"]
            for c in order
        ], [WIDTH * .49, WIDTH * .17, WIDTH * .17, WIDTH * .17]),
        Spacer(1, 5 * mm),
        figure("all_conditions.png", WIDTH * 4.6 / 9),
        p("All errors are illuminant recovery angles in degrees; lower is better. "
          "Worst 25% is the mean error of the highest-error quarter. Flat controls preserve "
          "image rays, RGB and missingness; they are not pure RGB-only architectures.", "SmallReport"),
        PageBreak(),
        p("EVIDENCE / ALL PREDECLARED COMPARISONS", "Kicker"),
        p("Read the paired differences", "TitleReport"),
        p("Each difference compares the same 324 out-of-fold captures. Negative values "
          "favor the first condition. Both primary intervals include zero."),
    ]
    short = {
        "camera_unitdepth minus native": "Normalized minus native",
        "camera_unitdepth minus camera_flat": "Normalized minus canonical flat",
        "camera_real minus native": "Canonical axes minus native",
        "camera_unitdepth minus camera_real": "Normalized minus canonical axes",
        "camera_flat minus native_flat": "Canonical flat minus native flat",
    }
    pair_rows = []
    for pair in pairs:
        lo, hi = pair["conditional_capture_bootstrap95"]
        pair_rows.append([short[pair["contrast"]], pair["role"].capitalize(),
                          f"{pair['mean_deg']:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    flow.extend([
        table(["Paired contrast", "Role", "Mean", "95% interval"], pair_rows,
              [WIDTH * .43, WIDTH * .15, WIDTH * .13, WIDTH * .29]),
        Spacer(1, 3 * mm),
        figure("paired_contrasts.png", WIDTH * 4.5 / 10),
        p("Conditional capture-bootstrap intervals describe these trained folds and captures. "
          "They do not measure training-seed or independent-room uncertainty. The two primary "
          "comparisons are not multiplicity adjusted.", "SmallReport"),
        p("Source-specific results retain the harm", "Subhead"),
        table(["Input representation", "NYU (n = 258)", "DIODE (n = 66)"], [
            [names[c], f"{final[(c, 'NYU')]['mean']:.4f}",
             f"{final[(c, 'DIODE')]['mean']:.4f}"] for c in order
        ], [WIDTH * .52, WIDTH * .24, WIDTH * .24]),
        Spacer(1, 2 * mm),
        p("The DIODE mean rises from 2.6175 to 3.3268 degrees after normalization. "
          "The canonical flat control also worsens versus the native flat control. "
          "These observations do not identify a physical cause or justify selecting another "
          "transform using the same test outcomes.", "SmallReport"),
        PageBreak(),
        p("REPRODUCIBILITY / INTERPRETATION / NEXT DECISION", "Kicker"),
        p("What this establishes", "TitleReport"),
        p("The experiment tests a representation hypothesis for <b>global illumination "
          "estimation from RGB-D</b>. It does not test monocular-depth accuracy. No reliability "
          "gate or new fusion architecture has been implemented."),
        p("The frozen comparison", "Subhead"),
        p("Canonical axes retain native scene scale. The normalized condition divides valid "
          "XYZ by the capture's median absolute depth; canonical flat divides each valid point "
          "by its own positive Z. RGB, labels and zero-geometry masks remain unchanged. "
          "Axis inference uses the released ordered 16 x 16 grid; it is not general calibration "
          "of an unordered point cloud."),
        p("All new models use the released sum head, 256 points, Adam at 0.0003, batch 16, "
          "the same augmentation and one predeclared seed per fold. Epoch 10,000 is primary. "
          "Best-test-fold checkpoints are separate optimistic references in summary.csv."),
        p("Audit and execution", "Subhead"),
        p("<b>PASS:</b> all 15 final checkpoints were re-inferred locally; 3,240 saved prediction "
          "rows were independently checked. The maximum independent angle discrepancy was "
          f"{audit['max_independent_atan2_discrepancy_deg']:.2e} degrees. Input, configuration, "
          "checkpoint and original execution-source identities were checked."),
        p("A known wrapper metadata error occurred after completed outputs were saved. "
          "The outputs were recovered and audited without retraining; the actual v1 wrapper "
          "is preserved in snapshots/run_modal_v1.py. Full details are in EXECUTION_NOTES.md."),
        p(f"New training used {audit['training_gpu_hours']:.3f} GPU-hours. The estimated training "
          f"resource allowance is ${audit['training_resource_planning_usd_upper']:.2f}, excluding "
          "smoke/build/startup and other overhead. This is not a provider invoice. The approved "
          "experiment cap was $8; completion records report the app stopped with no retry pending.",
          "SmallReport"),
        p("Limits and practical next step", "Subhead"),
        p("Prior benchmark outcomes motivated this study, so these folds are development "
          "evidence. One seed per fold and limited capture/room independence restrict the "
          "claim. Coordinate differences do not prove dataset corruption or physically wrong units."),
        p("The local ETH3D inventory verifies all 195 benchmark clouds/labels and the existing "
          "frozen depth checkpoint. Original full RGBs and a verified image-to-point mapping "
          "are missing, so valid predicted-depth inputs cannot yet be prepared. Obtain these "
          "prerequisites and freeze a separate protocol before training. The proposed comparison "
          "connects to monocular inference while retaining illuminant estimation as the endpoint. "
          "Reliability-aware fusion and later mixed illumination remain proposals; normalization "
          "should stay an ablation. See NEXT_STEPS.md and ETH3D_FEASIBILITY.md. No new training "
          "or paid compute was used to prepare this report."),
        p("Evidence and rebuild", "Subhead"),
        p("Relative to phase7_depth_format/: PROTOCOL.md; RESULTS.md; EXECUTION_NOTES.md; "
          "NEXT_STEPS.md; ETH3D_FEASIBILITY.md; results/nyu_frame_v1/{summary.csv, paired.json, audit.json}. "
          "The machine-readable report_manifest.json records source hashes. Rebuild locally "
          "with .venv/Scripts/python.exe phase7_depth_format/build_report.py.", "SmallReport"),
    ])

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d4dfe6"))
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont("Report", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 11.5 * mm, "Shirsha Mitra | PCCC representation diagnostic | 16 November 2026")
        canvas.drawRightString(A4[0] - 18 * mm, 11.5 * mm, f"{doc.page}")
        canvas.restoreState()

    PDF.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=21 * mm,
        title="PCCC Camera-Frame and Depth-Scale Experiment - 16 November 2026",
        author="Shirsha Mitra", subject="Completed audited research progress; no normalization benefit established",
    )
    document.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    sources = [HOME / name for name in [
        "PROTOCOL.md", "RESULTS.md", "EXECUTION_NOTES.md", "NEXT_STEPS.md", "ETH3D_FEASIBILITY.md", "analyze.py",
    ]] + [RESULTS / name for name in [
        "summary.json", "summary.csv", "paired.json", "audit.json",
        "figures/all_conditions.png", "figures/paired_contrasts.png",
    ]]
    manifest = {
        "report_date": "2026-11-16", "timezone": "Asia/Shanghai",
        "report": PDF.relative_to(ROOT).as_posix(), "report_sha256": sha(PDF),
        "builder": Path(__file__).relative_to(ROOT).as_posix(), "builder_sha256": sha(Path(__file__)),
        "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
        "training_or_cloud_launched": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(PDF), "source_manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    build()
