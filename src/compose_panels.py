"""Compose individual vector PDF plots into a single multi-panel figure.

Assembles one PDF per panel into a single vector PDF, row by row, and draws
bold lower-case panel letters (a, b, c, ...) in the top-left corner of each
panel. Rows may hold different numbers of panels, and each row can be
aligned left, centred or right within the figure width.

Nothing is rasterized: the source drawing operators are placed as XObjects,
so the output stays fully editable and text stays selectable.

Nature portfolio specification implemented here:
    - maximum width 180 mm (double column), 88 mm single column
    - panel letters in lower-case bold, positioned top-left
    - all panels placed at the *same* scale, so that a 7 pt label in a
      source file is still a 7 pt label in the composed figure

Requires: pymupdf (>= 1.23). numpy is optional, used only for auto-cropping.

Typical use
-----------
    $ python compose_panels.py

Author note: the single most common cause of a rejected multi-panel figure
is per-panel scaling, which silently turns uniform 7 pt labels into a mix
of 5 pt and 9 pt. Keep scale_mode="native" whenever the source panels were
generated at their final physical size.
"""

from __future__ import annotations

import string
from pathlib import Path

import pymupdf

MM = 72.0 / 25.4  # PDF points per millimetre


def ink_bbox(page: pymupdf.Page, dpi: int = 36, threshold: int = 250) -> pymupdf.Rect:
    """Return the tight bounding box of the drawn content of a PDF page.

    The page is rasterized at low resolution purely to *measure* where ink
    is; the returned rectangle is in PDF points and is used to crop the
    vector placement, so output quality is unaffected by `dpi`.

    Parameters
    ----------
    page : pymupdf.Page
        Page to measure.
    dpi : int, optional
        Resolution of the measurement raster. 36 is enough to locate
        content to within ~0.7 pt; raise it if panels have very thin
        hairlines near the edge. Default 36.
    threshold : int, optional
        Grey level (0-255) below which a pixel counts as ink. Default 250,
        which tolerates anti-aliasing and faint background shading.

    Returns
    -------
    pymupdf.Rect
        Tight bounding box in PDF points, in page coordinates. Falls back
        to `page.rect` if the page is blank or numpy is unavailable.

    Examples
    --------
    >>> doc = pymupdf.open("panel_a.pdf")          # doctest: +SKIP
    >>> bbox = ink_bbox(doc[0])                    # doctest: +SKIP
    >>> round(bbox.width) > 0                      # doctest: +SKIP
    True
    """
    try:
        import numpy as np
    except ImportError:
        return page.rect

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    mask = arr < threshold
    if not mask.any():
        return page.rect

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r = page.rect
    return pymupdf.Rect(
        r.x0 + cols[0] / zoom,
        r.y0 + rows[0] / zoom,
        r.x0 + (cols[-1] + 1) / zoom,
        r.y0 + (rows[-1] + 1) / zoom,
    )


def _as_rows(sources, ncols):
    """Normalize the `sources` argument into a list of rows of Paths.

    Parameters
    ----------
    sources : sequence
        Either a nested sequence (one inner sequence per row) or a flat
        sequence of paths, in which case `ncols` splits it into rows.
    ncols : int or None
        Number of columns used to chunk a flat `sources`. Ignored when
        `sources` is already nested.

    Returns
    -------
    list of list of pathlib.Path

    Raises
    ------
    ValueError
        If `sources` is empty, or is flat and `ncols` is not a positive int.

    Examples
    --------
    >>> _as_rows([["a.pdf", "b.pdf"], ["c.pdf"]], None)
    [[PosixPath('a.pdf'), PosixPath('b.pdf')], [PosixPath('c.pdf')]]
    >>> _as_rows(["a.pdf", "b.pdf", "c.pdf"], 2)
    [[PosixPath('a.pdf'), PosixPath('b.pdf')], [PosixPath('c.pdf')]]
    """
    if not sources:
        raise ValueError("sources is empty")

    nested = isinstance(sources[0], (list, tuple))
    if nested:
        rows = [[Path(p) for p in row] for row in sources]
    else:
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("a flat `sources` requires a positive integer `ncols`")
        flat = [Path(p) for p in sources]
        rows = [flat[i:i + ncols] for i in range(0, len(flat), ncols)]

    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("sources contains no panel")
    return rows


def _per_row(value, nrows, allowed, name):
    """Broadcast a scalar option to one value per row, or validate a sequence.

    Parameters
    ----------
    value : str or sequence of str
        Either a single value applying to every row, or one value per row.
    nrows : int
        Number of rows in the layout.
    allowed : set of str
        Accepted values.
    name : str
        Option name, used in error messages.

    Returns
    -------
    list of str
        One value per row.

    Raises
    ------
    ValueError
        If a value is not in `allowed`, or if a sequence has the wrong length.

    Examples
    --------
    >>> _per_row("center", 3, {"left", "center", "right"}, "align")
    ['center', 'center', 'center']
    >>> _per_row(["left", "right"], 2, {"left", "center", "right"}, "align")
    ['left', 'right']
    """
    if isinstance(value, str):
        out = [value] * nrows
    else:
        out = list(value)
        if len(out) != nrows:
            raise ValueError(f"{name}: expected {nrows} values, got {len(out)}")
    bad = set(out) - allowed
    if bad:
        raise ValueError(
            f"{name}: unknown value(s) {sorted(bad)}; allowed {sorted(allowed)}"
        )
    return out


def compose_panels(
    sources,
    output,
    ncols=None,
    width_mm=180.0,
    scale_mode="native",
    align="center",
    valign="top",
    hgap_mm=4.0,
    vgap_mm=4.0,
    margin_mm=1.0,
    label_size=8.0,
    label_font="Helvetica-Bold",
    label_fontfile=None,
    label_dx_mm=0.0,
    label_dy_mm=0.0,
    labels=None,
    autocrop=True,
):
    """Assemble single-panel PDFs into one multi-panel figure PDF.

    Panels are laid out one row at a time. Rows are independent: a row of
    two panels and a row of three are both allowed, and each row is placed
    within the figure width according to `align`. The figure width is set
    by the widest row (scale_mode="native") or by `width_mm`
    (scale_mode="fit"). Within a row, panels are stacked left to right,
    separated by `hgap_mm`, and aligned vertically according to `valign`.

    Parameters
    ----------
    sources : sequence
        Either a nested sequence, one inner sequence per row, e.g.
        ``[["a.pdf", "b.pdf"], ["c.pdf"], ["d.pdf", "e.pdf", "f.pdf"]]``,
        or a flat sequence of paths together with `ncols`. Panels are read
        in that order. The first page of each file is used.
    output : str or pathlib.Path
        Destination PDF path.
    ncols : int, optional
        Number of columns, used only to chunk a flat `sources` into rows.
        Ignored when `sources` is nested. Default None.
    width_mm : float, optional
        Total figure width in millimetres, used only when
        `scale_mode="fit"`. Nature allows 180 mm maximum (double column)
        and 88 mm for a single column. Default 180.0.
    scale_mode : {"native", "fit"}, optional
        "native" places every panel at 1:1 (no scaling at all) and derives
        the figure width from the panels themselves — use this when the
        sources were generated at their final physical size, as it is the
        only mode that preserves the source font sizes exactly.
        "fit" rescales the whole layout uniformly so the result is exactly
        `width_mm` wide; the scale factor is identical for all panels, so
        relative font sizes are preserved but absolute ones change.
        Default "native".
    align : {"left", "center", "right"} or sequence, optional
        Horizontal placement of each row within the figure width. Pass a
        single value for all rows, or one value per row, e.g.
        ``["left", "center", "left"]``. Default "center".
    valign : {"top", "center", "bottom"} or sequence, optional
        Vertical placement of a panel within its row, when panels of that
        row have unequal heights. Pass a single value or one per row.
        Default "top".
    hgap_mm, vgap_mm : float, optional
        Gaps between panels of a row and between rows, in millimetres.
        Default 4.0.
    margin_mm : float, optional
        Outer margin in millimetres. Keep small; journals crop anyway.
        Default 1.0.
    label_size : float, optional
        Panel letter size in points. Nature uses 8 pt bold. Default 8.0.
    label_font : str, optional
        Font name. With `label_fontfile` set, any name works and becomes
        the internal resource name; otherwise it must be a base-14 PDF
        font. Default "Helvetica-Bold".
    label_fontfile : str or pathlib.Path, optional
        Path to a bold sans-serif .ttf (Arial Bold, Helvetica Bold, or
        DejaVuSans-Bold as a fallback). Base-14 fonts are legal but are not
        embedded, and production preflight usually asks for embedded fonts,
        so set this for the final submission files. Default None.
    label_dx_mm, label_dy_mm : float, optional
        Offset of the letter from the panel's top-left ink corner, in
        millimetres. Positive dx moves right, positive dy moves down.
        Default 0.0.
    labels : sequence, optional
        Explicit panel labels, either flat in reading order or nested to
        mirror the row structure. Defaults to "a", "b", "c", ... across all
        rows. Pass an empty sequence to draw no labels, e.g. when the
        sources already carry their own.
    autocrop : bool, optional
        Crop each source to its ink bounding box before placement. This is
        what makes panels line up: matplotlib PDFs carry uneven white
        margins that otherwise offset every panel differently.
        Default True.

    Returns
    -------
    pathlib.Path
        Path to the written PDF.

    Raises
    ------
    FileNotFoundError
        If a source file does not exist.
    ValueError
        If `sources` is empty, if `scale_mode`, `align` or `valign` has an
        unknown value, or if a per-row sequence has the wrong length.

    Examples
    --------
    Three rows of 2, 1 and 3 panels, the single-panel row centred:

    >>> compose_panels(
    ...     [["fig4a.pdf", "fig4b.pdf"],
    ...      ["fig4c.pdf"],
    ...      ["fig4d.pdf", "fig4e.pdf", "fig4f.pdf"]],
    ...     "fig4.pdf",
    ...     align=["left", "center", "left"],
    ... )                                          # doctest: +SKIP
    PosixPath('fig4.pdf')

    A flat list chunked into rows of two, forced to 180 mm:

    >>> compose_panels(
    ...     ["a.pdf", "b.pdf", "c.pdf", "d.pdf"],
    ...     "fig2.pdf",
    ...     ncols=2,
    ...     scale_mode="fit",
    ...     width_mm=180,
    ... )                                          # doctest: +SKIP
    PosixPath('fig2.pdf')
    """
    rows = _as_rows(sources, ncols)
    nrows = len(rows)

    for row in rows:
        for s in row:
            if not s.exists():
                raise FileNotFoundError(s)
    if scale_mode not in ("native", "fit"):
        raise ValueError(f"unknown scale_mode: {scale_mode!r}")

    aligns = _per_row(align, nrows, {"left", "center", "right"}, "align")
    valigns = _per_row(valign, nrows, {"top", "center", "bottom"}, "valign")

    if labels is None:
        flat_labels = list(string.ascii_lowercase)
    elif labels and isinstance(labels[0], (list, tuple)):
        flat_labels = [lab for row in labels for lab in row]
    else:
        flat_labels = list(labels)

    # --- measure every panel -------------------------------------------------
    docs, boxes = [], []
    for row in rows:
        doc_row, box_row = [], []
        for s in row:
            doc = pymupdf.open(s)
            page = doc[0]
            box_row.append(ink_bbox(page) if autocrop else page.rect)
            doc_row.append(doc)
        docs.append(doc_row)
        boxes.append(box_row)

    hgap, vgap, margin = hgap_mm * MM, vgap_mm * MM, margin_mm * MM

    row_w = [sum(b.width for b in row) + hgap * (len(row) - 1) for row in boxes]
    row_h = [max(b.height for b in row) for row in boxes]
    content_w = max(row_w)
    content_h = sum(row_h) + vgap * (nrows - 1)

    scale = 1.0
    if scale_mode == "fit":
        scale = (width_mm * MM - 2 * margin) / content_w

    page_w = content_w * scale + 2 * margin
    page_h = content_h * scale + 2 * margin

    # --- place ---------------------------------------------------------------
    out = pymupdf.open()
    page = out.new_page(width=page_w, height=page_h)

    k = 0
    for r, row in enumerate(boxes):
        slack = (content_w - row_w[r]) * scale
        if aligns[r] == "left":
            x = margin
        elif aligns[r] == "right":
            x = margin + slack
        else:
            x = margin + slack / 2.0

        y_row = margin + scale * (sum(row_h[:r]) + vgap * r)

        for c, box in enumerate(row):
            w, h = box.width * scale, box.height * scale
            if valigns[r] == "center":
                y = y_row + (row_h[r] * scale - h) / 2.0
            elif valigns[r] == "bottom":
                y = y_row + (row_h[r] * scale - h)
            else:
                y = y_row

            page.show_pdf_page(
                pymupdf.Rect(x, y, x + w, y + h),
                docs[r][c],
                0,
                clip=box,
                keep_proportion=True,
            )

            if k < len(flat_labels):
                page.insert_text(
                    pymupdf.Point(
                        x + label_dx_mm * MM,
                        y + label_dy_mm * MM + label_size,
                    ),
                    flat_labels[k],
                    fontname=label_font,
                    fontfile=str(label_fontfile) if label_fontfile else None,
                    fontsize=label_size,
                )
            k += 1
            x += w + hgap * scale

    out.save(str(output), garbage=4, deflate=True)
    out.close()
    for doc_row in docs:
        for doc in doc_row:
            doc.close()

    shape = "+".join(str(len(row)) for row in rows)
    print(
        f"{output}: {page_w / MM:.1f} x {page_h / MM:.1f} mm, "
        f"rows {shape}, scale {scale:.3f}"
    )
    return Path(output)


# --------------------------------------------------------------------------
# rcParams to put at the top of every plotting script, so that panels come
# out at their final physical size with journal-compliant type.
#
#   import matplotlib.pyplot as plt
#   plt.rcParams.update(NATURE_RC)
#   fig, ax = plt.subplots(figsize=(88 / 25.4, 60 / 25.4))   # single column
#   ...
#   fig.savefig("panel_a.pdf", bbox_inches="tight", pad_inches=0.01)
#
# pdf.fonttype 42 is the important one: matplotlib defaults to Type 3, which
# production teams cannot edit and which some preflight tools reject.
# --------------------------------------------------------------------------
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
}


if __name__ == "__main__":
    compose_panels(
        [
            ["panels/a.pdf", "panels/b.pdf"],
            ["panels/c.pdf"],
            ["panels/d.pdf", "panels/e.pdf", "panels/f.pdf"],
        ],
        "figure.pdf",
        align=["left", "center", "left"],
        valign="top",
        scale_mode="native",
        hgap_mm=4,
        vgap_mm=4,
    )
