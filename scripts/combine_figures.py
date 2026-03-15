#!/usr/bin/env python3
"""
Combine three PDF plots from each results subfolder into a single
vertically-stacked PDF (and a high-resolution PNG) using PyMuPDF.

Each subfigure is first cropped to its tight content bounding box
(removing matplotlib's internal whitespace/margins), then rescaled
to fill a common width.  The three cropped panels are stacked so the
combined figure is approximately 7 x 7 inches.

Outputs are saved to results/combined_figures/.
"""

import os
import glob
import pymupdf  # PyMuPDF >= 1.24

# ── paths ────────────────────────────────────────────────────────────
BASE = "/Users/wanda/QSTC PROJECT/PaperBanana/results"
OUT_DIR = os.path.join(BASE, "combined_figures")
os.makedirs(OUT_DIR, exist_ok=True)

FOLDERS = [
    "best_fidelity_noise",
    "best_iterations_noise",
    "best_tokens_noise",
    "noise_level_fidelity",
    "noise_level_iterations",
    "noise_level_token",
    "prompt_fidelity",
    "prompt_iterations",
    "prompt_token",
]

# ── layout parameters ────────────────────────────────────────────────
TARGET_WIDTH_IN = 7.0
TARGET_WIDTH = TARGET_WIDTH_IN * 72  # 504 pt
MAX_COMBINED_HEIGHT_IN = 7.0
GAP = 6  # vertical gap (pt) between panels
PAD = 2  # padding (pt) kept around cropped content
DPI = 300


# ── helpers ──────────────────────────────────────────────────────────
def content_bbox(page, threshold=248):
    """Return the tight bounding box of non-white pixels (in PDF coords).

    Renders the page at 2x, scans rows/cols for non-white content,
    then maps back to page coordinates.  A small PAD is added.
    """
    zoom = 2.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    w, h, stride = pix.width, pix.height, pix.stride
    samples = pix.samples  # raw bytes, 3 bytes per pixel (RGB)
    n = pix.n  # number of components per pixel

    min_x, min_y = w, h
    max_x, max_y = 0, 0

    for y in range(h):
        row_start = y * stride
        for x in range(w):
            idx = row_start + x * n
            if (
                samples[idx] < threshold
                or samples[idx + 1] < threshold
                or samples[idx + 2] < threshold
            ):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    # Map pixel coords -> PDF coords and add padding
    rect = page.rect
    x0 = max(rect.x0, min_x / zoom - PAD)
    y0 = max(rect.y0, min_y / zoom - PAD)
    x1 = min(rect.x1, max_x / zoom + PAD)
    y1 = min(rect.y1, max_y / zoom + PAD)
    return pymupdf.Rect(x0, y0, x1, y1)


def combine_pdfs_vertically(folder_name: str) -> None:
    folder_path = os.path.join(BASE, folder_name)
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))

    if len(pdf_files) != 3:
        print(f"  [SKIP] {folder_name}: expected 3 PDFs, found {len(pdf_files)}")
        return

    print(f"  Processing {folder_name} …")

    src_docs = [pymupdf.open(p) for p in pdf_files]
    src_pages = [doc[0] for doc in src_docs]

    # ── crop each source page to its content bbox ────────────────────
    crop_rects = [content_bbox(p) for p in src_pages]

    for f, cr in zip(pdf_files, crop_rects):
        print(
            f"    • {os.path.basename(f):50s}  "
            f"crop {cr.width / 72:.2f} x {cr.height / 72:.2f} in"
        )

    # ── compute target panel sizes ───────────────────────────────────
    # Scale each cropped panel to TARGET_WIDTH, preserving aspect ratio
    proportional_heights = [cr.height * (TARGET_WIDTH / cr.width) for cr in crop_rects]

    total_gaps = GAP * (len(crop_rects) - 1)
    max_content = MAX_COMBINED_HEIGHT_IN * 72 - total_gaps
    total_prop = sum(proportional_heights)

    compress = min(1.0, max_content / total_prop)
    panel_heights = [h * compress for h in proportional_heights]

    # ── build the combined canvas ────────────────────────────────────
    total_height = sum(panel_heights) + total_gaps

    dst_doc = pymupdf.open()
    dst_page = dst_doc.new_page(width=TARGET_WIDTH, height=total_height)

    y_offset = 0.0
    for src_page, cr, ph in zip(src_pages, crop_rects, panel_heights):
        target_rect = pymupdf.Rect(0, y_offset, TARGET_WIDTH, y_offset + ph)
        # clip= restricts which part of the source page is rendered
        dst_page.show_pdf_page(target_rect, src_page.parent, src_page.number, clip=cr)
        y_offset += ph + GAP

    # ── save ─────────────────────────────────────────────────────────
    out_pdf = os.path.join(OUT_DIR, f"{folder_name}_combined.pdf")
    dst_doc.save(out_pdf, deflate=True)

    h_in = total_height / 72
    print(
        f"    => combined: {TARGET_WIDTH_IN:.1f} x {h_in:.2f} in  "
        f"(compress={compress:.2f})"
    )
    print(f"    -> {out_pdf}")

    zoom = DPI / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = dst_page.get_pixmap(matrix=mat)
    out_png = os.path.join(OUT_DIR, f"{folder_name}_combined.png")
    pix.save(out_png)
    print(f"    -> {out_png}")

    dst_doc.close()
    for d in src_docs:
        d.close()


def main():
    print(f"Output directory: {OUT_DIR}")
    print(
        f"Target: {TARGET_WIDTH_IN} x ~{MAX_COMBINED_HEIGHT_IN} in  |  "
        f"padding={PAD}pt  gap={GAP}pt\n"
    )
    for folder in FOLDERS:
        combine_pdfs_vertically(folder)
    print("\nDone – all folders processed.")


if __name__ == "__main__":
    main()
