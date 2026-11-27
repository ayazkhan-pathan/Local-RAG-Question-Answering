"""Draw the README figures from the files in results/.

    pipeline.png             the pipeline, with one question traced through it
    book-map.png             chunk embeddings projected to 2-D, coloured by chapter
    retrieval-grid.png       hit@3 for every model, chunking and query-instruction setting
    closed-book-vs-rag.png   answer accuracy of the local model with and without retrieval
    threshold.png            best similarity of book and off-topic questions, with the cut-off

Usage:
    python scripts/make_figures.py                       # all figures into docs/images/
    python scripts/make_figures.py --only pipeline --example q01
    python scripts/make_figures.py --results runs/try/results --out runs/try/images

Figures are drawn at 1,760 px wide, twice the width GitHub shows them at. The illustration in
pipeline.png is John Tenniel's White Rabbit (1865, public domain), from Project Gutenberg eBook
#114.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from rag_local.config import DATA_DIR, QUESTIONS, Settings  # noqa: E402
from rag_local.documents import load_books  # noqa: E402
from rag_local.evaluation import find_quote, load_questions  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#eb6834"
BLUE = "#2a78d6"
CONTEXT = "#c3c2b7"
CHAPTER_RAMP = LinearSegmentedColormap.from_list("chapters", ["#a9c8ee", "#2a78d6", "#0f2f5c"])

DPI = 200
WIDTH = 8.8  # inches: 1,760 px at 200 dpi, shown at 880 px
TITLE, SUBTITLE, BODY, SMALL = 13, 10, 9.5, 9
DIMENSIONS = {"bge-small": 384, "bge-base": 768, "bge-large": 1024}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
ILLUSTRATION = Path("docs/assets/tenniel-white-rabbit.png")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": BODY,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------- data


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def pick_example(answers: list[dict], wanted: str | None, prefer: str = "q01") -> str:
    """The question to trace: as asked; else ``prefer`` (the White Rabbit's watch, to match the
    illustration) or the first question that RAG answers correctly with the answer passage
    retrieved while the closed-book answer is wrong."""
    if wanted:
        return wanted
    closed = {r["id"]: r for r in answers if r["mode"] == "closed_book"}
    good = [
        r["id"]
        for r in answers
        if r["mode"] == "rag"
        and r.get("correct")
        and r["gold_retrieved"]
        and not closed[r["id"]]["correct"]
    ]
    if prefer in good:
        return prefer
    return good[0] if good else next(r["id"] for r in answers if r["mode"] == "rag")


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {path.as_posix()}")
    return path


def heading(fig, x: float, title: str, subtitle: str | None = None, top: float = 0.0) -> None:
    """A left-aligned title and subtitle, ``top`` inches below the top edge."""
    height = fig.get_figheight()
    fig.text(x, 1 - (top + 0.30) / height, title, fontsize=TITLE, fontweight="bold", va="baseline")
    if subtitle:
        fig.text(
            x, 1 - (top + 0.58) / height, subtitle, fontsize=SUBTITLE, color=INK2, va="baseline"
        )


def tidy(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- pipeline (hero)


class Canvas:
    """A full-figure axes in output pixels (y down), with word-wrapped rich text."""

    def __init__(self, height_px: int):
        self.w, self.h = int(WIDTH * DPI), height_px
        self.fig = plt.figure(figsize=(WIDTH, height_px / DPI), dpi=DPI)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(self.h, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()

    def width(self, text: str, **style) -> float:
        artist = self.ax.text(0, 0, text, **style)
        extent = artist.get_window_extent(self.renderer).width
        artist.remove()
        return extent

    def text(self, x, y, text, **style):
        return self.ax.text(x, y, text, va="baseline", **style)

    def rich(self, x, y, width, segments, size, line_height, max_lines=None) -> float:
        """Draw (text, marked) segments word by word; marked words get a highlighter band.
        A segment that does not start with a space joins the previous word. Returns the
        baseline of the last line."""
        space = self.width("x x", fontsize=size) - self.width("xx", fontsize=size)
        cx, cy, line, previous = x, y, 1, " "
        for text, marked in segments:
            glue = bool(text) and not text[0].isspace() and not previous[-1:].isspace()
            for n, word in enumerate(text.split()):
                style = {
                    "fontsize": size,
                    "color": INK if marked else INK2,
                    "fontweight": "bold" if marked else "normal",
                }
                if n == 0 and glue:
                    cx -= space
                w = self.width(word, **style)
                if cx > x and cx + w > x + width:
                    line += 1
                    if max_lines and line > max_lines:
                        return cy
                    cx, cy = x, cy + line_height
                if marked:
                    pad = space / 2 + 1
                    self.ax.add_patch(
                        Rectangle(
                            (cx - pad, cy - 0.78 * line_height),
                            w + 2 * pad,
                            0.98 * line_height,
                            facecolor=ACCENT,
                            alpha=0.22,
                            edgecolor="none",
                            zorder=1,
                        )
                    )
                self.text(cx, cy, word, zorder=2, **style)
                cx += w + space
            previous = text or previous
        return cy

    def finish(self, height_px: int) -> None:
        """Trim or extend the canvas to ``height_px`` once the layout is known."""
        self.h = int(height_px)
        self.fig.set_size_inches(WIDTH, self.h / DPI)
        self.ax.set_ylim(self.h, 0)


def glyph(ax, kind: str, x: float, y: float, s: float, colour: str) -> None:
    """Small white pictograms drawn inside the station circles (s = half size in px)."""
    white = {"facecolor": "white", "edgecolor": "none", "zorder": 5}
    line = {"color": "white", "linewidth": 1.6, "solid_capstyle": "round", "zorder": 5}
    if kind == "book":
        ax.add_patch(Rectangle((x - 0.62 * s, y - 0.62 * s), 0.56 * s, 1.24 * s, **white))
        ax.add_patch(Rectangle((x + 0.06 * s, y - 0.62 * s), 0.56 * s, 1.24 * s, **white))
    elif kind == "chunks":
        for dy in (-0.45, 0.0, 0.45):
            ax.add_patch(
                Rectangle((x - 0.6 * s, y + dy * s - 0.14 * s), 1.2 * s, 0.28 * s, **white)
            )
    elif kind == "vectors":
        for i, dx in enumerate((-0.45, 0.0, 0.45)):
            for j, dy in enumerate((-0.45, 0.0, 0.45)):
                r = 0.13 * s if (i + j) % 2 else 0.2 * s
                ax.add_patch(Circle((x + dx * s, y + dy * s), r, **white))
    elif kind == "store":
        for dy in (-0.42, 0.0, 0.42):
            ax.add_patch(
                Rectangle((x - 0.55 * s, y + dy * s - 0.17 * s), 1.1 * s, 0.34 * s, **white)
            )
    elif kind == "topk":
        for dx, h in ((-0.45, 0.9), (0.0, 1.25), (0.45, 0.7)):
            ax.add_patch(
                Rectangle((x + dx * s - 0.14 * s, y + 0.6 * s - h * s), 0.28 * s, h * s, **white)
            )
    elif kind == "prompt":
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.62 * s, y - 0.5 * s),
                1.24 * s,
                0.85 * s,
                boxstyle=f"round,pad=0,rounding_size={0.2 * s}",
                **white,
            )
        )
        ax.add_patch(
            Polygon(
                [(x - 0.3 * s, y + 0.3 * s), (x - 0.45 * s, y + 0.7 * s), (x, y + 0.3 * s)],
                closed=True,
                **white,
            )
        )
        for dy in (-0.22, 0.05):
            ax.plot(
                [x - 0.38 * s, x + 0.38 * s],
                [y + dy * s, y + dy * s],
                color=colour,
                linewidth=1.2,
                zorder=6,
            )
    elif kind == "model":
        ax.add_patch(Rectangle((x - 0.45 * s, y - 0.45 * s), 0.9 * s, 0.9 * s, **white))
        for d in (-0.25, 0.0, 0.25):
            for x0, y0, x1, y1 in (
                (x + d * s, y - 0.45 * s, x + d * s, y - 0.68 * s),
                (x + d * s, y + 0.45 * s, x + d * s, y + 0.68 * s),
                (x - 0.45 * s, y + d * s, x - 0.68 * s, y + d * s),
                (x + 0.45 * s, y + d * s, x + 0.68 * s, y + d * s),
            ):
                ax.plot([x0, x1], [y0, y1], **line)
    elif kind == "answer":
        ax.plot(
            [x - 0.5 * s, x - 0.12 * s, x + 0.55 * s],
            [y, y + 0.4 * s, y - 0.45 * s],
            color="white",
            linewidth=2.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )


def excerpt_segments(text: str, quote: str, before: int = 150, after: int = 170) -> list:
    """The passage around the gold quote, with the quote itself marked."""
    span = find_quote(text, quote)[0]
    start = max(0, span[0] - before)
    end = min(len(text), span[1] + after)

    def clean(part: str) -> str:
        inner = " ".join(part.replace("_", "").split())
        lead = " " if part[:1].isspace() else ""
        trail = " " if part[-1:].isspace() else ""
        return lead + inner + trail if inner else lead

    head, tail = clean(text[start : span[0]]), clean(text[span[1] : end])
    if start > 0:
        head = "... " + head.strip().split(" ", 1)[-1] + (" " if head.endswith(" ") else "")
    if end < len(text):
        tail = tail.rstrip().rsplit(" ", 1)[0] + " ..."
    return [(head, False), (clean(text[span[0] : span[1]]).strip(), True), (tail, False)]


def figure_pipeline(results: Path, out: Path, data: Path, questions_path: Path, example: str):
    summary = json.loads((results / "answers_summary.json").read_text())
    settings = Settings(**summary["retrieval"])
    answers = read_jsonl(results / "answers.jsonl")
    questions = {q.id: q for q in load_questions(questions_path)}
    book = load_books(data)[0]
    question = questions[example]
    rag = next(r for r in answers if r["id"] == example and r["mode"] == "rag")
    closed = next(r for r in answers if r["id"] == example and r["mode"] == "closed_book")
    model = summary["model"].split("/")[-1]

    c = Canvas(1400)
    ax, left, right = c.ax, 56, c.w - 56
    c.text(left, 70, "From book to answer, on a laptop", fontsize=TITLE, fontweight="bold")
    c.text(
        left,
        118,
        f"The book is split and indexed once. Each question pulls its "
        f"{settings.k} closest passages into the prompt of a small local model.",
        fontsize=SUBTITLE,
        color=INK2,
    )

    kept = f"{(book.end - book.start) / 1000:.0f}k chars"
    dimension = DIMENSIONS.get(settings.embedding_model)
    numbers = f"{dimension:,} numbers" if dimension else "one vector"
    k, cut = settings.k, settings.threshold
    stations = [
        ("book", "Book", f"{len(book.chapter_starts)} chapters", kept),
        (
            "chunks",
            "Chunks",
            f"{summary['chunks']} chunks,",
            f"≤ {settings.chunk_size:,} chars each",
        ),
        ("vectors", "Embeddings", settings.embedding_model, numbers),
        ("store", "Chroma", "vector index,", "cosine space"),
        ("topk", f"Top {k}", f"{k} closest chunks", f"if score ≥ {cut:.2f}"),
        ("prompt", "Prompt", "passages", "+ question"),
        ("model", "Local LLM", model.replace("-Instruct", ""), "greedy, on device"),
        ("answer", "Answer", "with sources,", "or a refusal"),
    ]
    step = (right - left) / len(stations)
    xs = [left + step * (i + 0.5) for i in range(len(stations))]
    y, r = 268, 34
    ax.plot([xs[0], xs[3]], [y, y], color=INK2, linewidth=2.2, zorder=2)
    ax.plot([xs[3], xs[-1]], [y, y], color=ACCENT, linewidth=2.2, zorder=2)
    for label, x0, x1 in (("INDEXING, ONCE", xs[0], xs[3]), ("FOR EACH QUESTION", xs[4], xs[7])):
        ax.plot([x0 - r, x1 + r], [192, 192], color=GRID, linewidth=1.2)
        c.text(x0 - r, 178, label, fontsize=SMALL, color=MUTED, fontweight="bold")
    for i, ((kind, name, line1, line2), x) in enumerate(zip(stations, xs)):
        colour = INK2 if i < 4 else ACCENT
        ax.add_patch(Circle((x, y), r, facecolor=colour, edgecolor=SURFACE, linewidth=3, zorder=3))
        glyph(ax, kind, x, y, 0.62 * r, colour)
        c.text(x, y + r + 46, name, fontsize=BODY + 0.5, fontweight="bold", ha="center")
        c.text(x, y + r + 84, line1, fontsize=SMALL, color=INK2, ha="center")
        c.text(x, y + r + 118, line2, fontsize=SMALL, color=INK2, ha="center")

    top = 480
    ax.plot([left, right], [top - 30, top - 30], color=GRID, linewidth=1.2)
    image = plt.imread(ILLUSTRATION)
    img_h = 440
    img_w = img_h * image.shape[1] / image.shape[0]
    ax.imshow(
        image,
        extent=(left, left + img_w, top + img_h + 14, top + 14),
        zorder=1,
        interpolation="lanczos",
    )
    c.text(left, top + img_h + 62, "John Tenniel, 1865", fontsize=SMALL, color=MUTED)

    x0 = left + img_w + 70
    width = right - x0
    c.text(x0, top + 26, "QUESTION", fontsize=SMALL, color=MUTED, fontweight="bold")
    c.text(x0, top + 78, question.question, fontsize=BODY + 1.5)
    gold_rank = next(
        (i for i, hit in enumerate(rag["retrieved"], 1) if hit["id"] in rag["gold"]),
        None,
    )
    hit = rag["retrieved"][(gold_rank or 1) - 1]
    where = f"PASSAGE {gold_rank or 1} OF {settings.k}  ·  CHAPTER {ROMAN[hit['chapter']]}"
    c.text(
        x0,
        top + 150,
        f"{where}  ·  SIMILARITY {hit['score']:.2f}",
        fontsize=SMALL,
        color=MUTED,
        fontweight="bold",
    )
    start, end = (int(v) for v in hit["id"].rsplit(":", 1)[1].split("-"))
    chunk_text = book.text[start:end]
    last = c.rich(x0, top + 200, width, excerpt_segments(chunk_text, question.quote), BODY, 42, 4)
    y_answer = last + 82
    c.text(
        x0, y_answer, f"ANSWER FROM {model.upper()}", fontsize=SMALL, color=MUTED, fontweight="bold"
    )
    answer_text = " ".join(rag["output"].split())
    last = c.rich(x0, y_answer + 50, width, [(answer_text, False)], BODY + 1.5, 44, 3)
    closed_text = " ".join(closed["output"].split())
    if len(closed_text) > 170:
        closed_text = closed_text[:170].rsplit(" ", 1)[0] + " ..."
    verdict = "right" if closed["correct"] else "wrong"
    y_closed = last + 70
    c.text(
        x0,
        y_closed,
        f"SAME MODEL WITHOUT THE PASSAGES ({verdict.upper()})",
        fontsize=SMALL,
        color=MUTED,
        fontweight="bold",
    )
    last = c.rich(x0, y_closed + 44, width, [(closed_text, False)], SMALL + 0.5, 38, 2)
    c.finish(max(last, top + img_h + 62) + 40)
    return save(c.fig, out, "pipeline.png")


# ---------------------------------------------------------------- book map


def figure_book_map(results: Path, out: Path, questions_path: Path, example: str):
    rows = read_csv(results / "book_map.csv")
    best = json.loads((results / "retrieval_best.json").read_text())
    setting, k = best["best"]["setting"], best["settings"]["k"]
    questions = {q.id: q for q in load_questions(questions_path)}
    chunks = [r for r in rows if r["kind"] == "chunk"]
    point = next(r for r in rows if r["kind"] == "question" and r["id"] == example)
    per_q = {
        r["id"]: r for r in read_csv(results / "retrieval_questions.csv") if r["setting"] == setting
    }
    top = per_q[example]["top_ids"].split()[:k]
    gold = set(per_q[example]["gold"].split())

    x = np.array([float(r["x"]) for r in chunks])
    y = np.array([float(r["y"]) for r in chunks])
    chapter = np.array([int(r["chapter"]) for r in chunks])
    fig = plt.figure(figsize=(WIDTH, 6.0), dpi=DPI)
    ax = fig.add_axes((0.045, 0.1, 0.93, 0.7))
    body = chapter > 0
    ax.scatter(
        x[body],
        y[body],
        s=40,
        c=CHAPTER_RAMP((chapter[body] - 1) / 11),
        linewidths=0.9,
        edgecolors=SURFACE,
        zorder=2,
    )
    ax.scatter(x[~body], y[~body], s=40, c=CONTEXT, linewidths=0.9, edgecolors=SURFACE, zorder=2)
    halo = [patheffects.withStroke(linewidth=3.5, foreground=SURFACE)]
    placed: list[tuple[float, float]] = []
    for number in range(1, 13):
        mask = chapter == number
        if not mask.any():
            continue
        lx, ly = float(np.median(x[mask])), float(np.median(y[mask]))
        while any(
            abs(lx - px) < 0.05 * np.ptp(x) and abs(ly - py) < 0.04 * np.ptp(y) for px, py in placed
        ):
            ly -= 0.045 * np.ptp(y)
        placed.append((lx, ly))
        ax.text(
            lx,
            ly,
            ROMAN[number],
            fontsize=BODY + 1,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            path_effects=halo,
            zorder=6,
        )
    qx, qy = float(point["x"]), float(point["y"])
    where = {r["id"]: (float(r["x"]), float(r["y"])) for r in chunks}
    for rank, cid in enumerate(top, 1):
        cx, cy = where[cid]
        ax.plot([qx, cx], [qy, cy], color=ACCENT, linewidth=1.4, zorder=3)
        ax.scatter([cx], [cy], s=170, facecolors="none", edgecolors=ACCENT, linewidths=2, zorder=4)
        tag = f"{rank}, holds the answer" if cid in gold else f"{rank}"
        # Label on the far side from the question, so the line to the star never crosses it.
        right, up = cx >= qx, cy >= qy
        ax.annotate(
            tag,
            (cx, cy),
            xytext=(11 if right else -11, 6 if up else -6),
            textcoords="offset points",
            fontsize=SMALL,
            color=INK,
            ha="left" if right else "right",
            va="bottom" if up else "top",
            path_effects=halo,
            zorder=6,
        )
    ax.scatter(
        [qx], [qy], s=300, marker="*", color=ACCENT, edgecolors=SURFACE, linewidths=1.5, zorder=5
    )
    ax.annotate(
        "question",
        (qx, qy),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        fontsize=SMALL,
        color=INK,
        fontweight="bold",
        path_effects=halo,
        zorder=6,
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    pad_x, pad_y = 0.04 * np.ptp(x), 0.06 * np.ptp(y)
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    key = fig.add_axes((0.80, 0.815, 0.16, 0.022))
    key.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CHAPTER_RAMP)
    key.set_xticks([0, 255])
    key.set_xticklabels(["chapter I", "XII"], fontsize=SMALL)
    key.set_yticks([])
    for side in key.spines.values():
        side.set_visible(False)

    same, chance = best["nearest_same_chapter"], best["nearest_same_chapter_by_chance"]
    model = best["settings"]["embedding_model"]
    heading(
        fig,
        0.045,
        "Most passages are closest to others from their own chapter",
        f"For {pct(same)} of chunks, the most similar chunk they do not overlap is in the same "
        f"chapter ({pct(chance)} by chance).",
    )
    question = textwrap.shorten(questions[example].question, 110, placeholder=" ...")
    caption = (
        f"{model} embeddings projected to 2-D. The star is the question “{question}”, "
        f"linked to the {k} chunks it retrieves."
    )
    fig.text(
        0.045,
        0.055,
        textwrap.fill(caption, 128),
        fontsize=SMALL,
        color=INK2,
        va="top",
        linespacing=1.5,
    )
    return save(fig, out, "book-map.png")


# ---------------------------------------------------------------- retrieval grid


def figure_grid(results: Path, out: Path):
    rows = read_csv(results / "retrieval_grid.csv")
    original = next((r for r in rows if r["boilerplate"] == "True"), None)
    rows = [r for r in rows if r["boilerplate"] == "False"]
    models = [m for m in DIMENSIONS if any(r["model"] == m for r in rows)]
    chunkings = sorted({(int(r["chunk_size"]), int(r["chunk_overlap"])) for r in rows})
    best = next(r for r in rows if r["best"] == "True")
    values = [float(r["hit@3"]) for r in rows] + ([float(original["hit@3"])] if original else [])
    lo = max(0.0, np.floor(min(values) * 10) / 10 - 0.03)
    hi = min(1.0, np.ceil(max(values) * 10) / 10 + 0.03)

    fig = plt.figure(figsize=(WIDTH, 4.4), dpi=DPI)
    left, right, gap = 0.135, 0.975, 0.035
    panel = (right - left - gap * (len(models) - 1)) / len(models)
    for i, model in enumerate(models):
        ax = fig.add_axes((left + i * (panel + gap), 0.15, panel, 0.56))
        tidy(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(chunkings) - 0.45, -0.7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_yticks(range(len(chunkings)))
        ax.set_yticklabels([f"{s:,} / {o}" for s, o in chunkings] if i == 0 else [])
        ax.set_title(model, loc="left", fontsize=BODY, fontweight="bold", pad=4)
        for j, (size, overlap) in enumerate(chunkings):
            pair = {
                r["instruction"] == "True": r
                for r in rows
                if r["model"] == model
                and (int(r["chunk_size"]), int(r["chunk_overlap"])) == (size, overlap)
            }
            if len(pair) < 2:
                continue
            plain, instructed = float(pair[False]["hit@3"]), float(pair[True]["hit@3"])
            ax.plot([plain, instructed], [j, j], color=GRID, linewidth=3, zorder=2)
            for flag, value in ((True, instructed), (False, plain)):
                colour = ACCENT if pair[flag] is best else INK2
                if flag:
                    size = 64 if pair[flag] is best else 34
                    ax.scatter([value], [j], s=size, color=colour, linewidths=0, zorder=4)
                else:
                    ax.scatter(
                        [value],
                        [j],
                        s=110,
                        facecolors="none",
                        edgecolors=colour,
                        linewidths=1.5,
                        zorder=5,
                    )
                if pair[flag] is best:
                    ax.annotate(
                        f"best, {pct(value)}",
                        (value, j),
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha="center",
                        fontsize=SMALL,
                        fontweight="bold",
                    )
        if model == "bge-large" and original:
            value = float(original["hit@3"])
            ax.axvline(value, color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
            ax.annotate(
                "first version",
                (value, len(chunkings) - 0.45),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=SMALL,
                color=INK2,
            )
    fig.text(
        left,
        0.035,
        "questions whose answer passage is among the 3 retrieved",
        fontsize=SMALL,
        color=INK2,
    )
    fig.text(
        0.02, 0.43, "chunk size / overlap", rotation=90, fontsize=SMALL, color=INK2, va="center"
    )
    label = "with" if best["instruction"] == "True" else "without"
    heading(
        fig,
        left,
        f"{best['model']} on {int(best['chunk_size']):,}/{best['chunk_overlap']} chunks, "
        f"{label} the instruction, finds the passage most often",
        "Hit@3 on 40 questions. Ring: plain question. Dot: with the BGE query instruction.",
    )
    return save(fig, out, "retrieval-grid.png")


# ---------------------------------------------------------------- closed book vs RAG


def figure_accuracy(results: Path, out: Path):
    summary = json.loads((results / "answers_summary.json").read_text())
    n = summary["questions"]
    model = summary["model"].split("/")[-1].replace("-Instruct", "")
    bars = [
        ("Closed book: the question alone", summary["closed_book_correct"], CONTEXT),
        (
            f"RAG: the question with the top {summary['retrieval']['k']} passages",
            summary["rag_correct"],
            ACCENT,
        ),
    ]
    fig = plt.figure(figsize=(WIDTH, 3.0), dpi=DPI)
    left = 0.045
    ax = fig.add_axes((left, 0.16, 0.91, 0.5))
    tidy(ax)
    for i, (label, value, colour) in enumerate(bars):
        ax.barh(i, value / n, height=0.34, color=colour, zorder=3)
        ax.text(0, i - 0.3, label, fontsize=BODY, color=INK, va="bottom")
        ax.text(
            value / n + 0.01,
            i,
            f"{value} of {n}  ({pct(value / n)})",
            va="center",
            fontsize=BODY,
            fontweight="bold" if i else "normal",
        )
    ax.set_yticks([])
    ax.set_ylim(len(bars) - 0.6, -0.75)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    heading(
        fig,
        left,
        f"Retrieval lifts {model} from {summary['closed_book_correct']} to "
        f"{summary['rag_correct']} correct answers out of {n}",
        f"Same model, same questions, greedy decoding. The passage with the answer was among "
        f"those retrieved for {summary['rag_gold_retrieved']} of {n}.",
    )
    return save(fig, out, "closed-book-vs-rag.png")


# ---------------------------------------------------------------- threshold


def swarm(values: np.ndarray, gap: float) -> np.ndarray:
    """Row offsets (in multiples of one dot) that stop dots ``gap`` apart in x from touching."""
    offsets = np.zeros(len(values))
    placed: list[tuple[float, float]] = []
    for i in np.argsort(values):
        for step in range(40):
            candidate = (step + 1) // 2 * (1 if step % 2 else -1)
            if all(abs(values[i] - px) >= gap or candidate != py for px, py in placed):
                offsets[i] = candidate
                break
        placed.append((values[i], offsets[i]))
    return offsets


def figure_threshold(results: Path, out: Path):
    best = json.loads((results / "retrieval_best.json").read_text())
    setting, threshold = best["best"]["setting"], float(best["settings"]["threshold"])
    rows = [r for r in read_csv(results / "retrieval_questions.csv") if r["setting"] == setting]
    on = np.array([float(r["best_score"]) for r in rows if r["off_topic"] == "False"])
    off = np.array([float(r["best_score"]) for r in rows if r["off_topic"] == "True"])
    lo = np.floor(min(on.min(), off.min(), threshold) * 20) / 20 - 0.01
    hi = np.ceil(max(on.max(), off.max()) * 20) / 20 + 0.01

    fig = plt.figure(figsize=(WIDTH, 3.6), dpi=DPI)
    left = 0.045
    ax = fig.add_axes((left, 0.16, 0.91, 0.56))
    tidy(ax)
    ax.set_xlim(lo, hi)
    gap = (hi - lo) * 0.012
    rows_y = {0: 0.0, 1: 1.25}
    passed = []
    for i, (values, colour, name) in enumerate(
        ((on, BLUE, "Book questions"), (off, ACCENT, "Off-topic questions"))
    ):
        dy = swarm(values, gap) * 0.15
        ax.scatter(
            values, rows_y[i] + dy, s=46, color=colour, edgecolors=SURFACE, linewidths=1, zorder=3
        )
        count = int((values >= threshold).sum())
        passed.append(count)
        ax.text(
            lo,
            rows_y[i] - 0.5,
            f"{name}: {count} of {len(values)} at or above the cut-off",
            fontsize=BODY,
            color=INK,
            va="bottom",
            zorder=4,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 4},
        )
    ax.axvline(threshold, color=INK, linewidth=1.2, zorder=2)
    ax.annotate(
        f"cut-off {threshold:.2f}",
        (threshold, 1.75),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=SMALL,
        va="bottom",
    )
    ax.set_yticks([])
    ax.set_ylim(1.85, -0.75)
    ax.set_xlabel("cosine similarity of the closest chunk", fontsize=SMALL)
    if off.max() < on.min():
        title = "Every off-topic question scores below every book question"
        subtitle = (
            f"Best similarity per question. On these {len(on) + len(off)} questions any cut-off "
            f"between {off.max():.2f} and {on.min():.2f} would separate the two groups."
        )
    else:
        title = f"At {threshold:.2f}, {passed[1]} of {len(off)} off-topic questions still pass"
        subtitle = "Best similarity per question; below the cut-off the pipeline refuses."
    heading(fig, left, title, subtitle)
    return save(fig, out, "threshold.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--questions", default=QUESTIONS)
    parser.add_argument("--example", help="question id to trace (default: picked from results)")
    names = ["pipeline", "book-map", "grid", "accuracy", "threshold"]
    parser.add_argument("--only", nargs="+", choices=names)
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    wanted = set(args.only or names)
    example = args.example
    if "pipeline" in wanted or "book-map" in wanted:
        example = pick_example(read_jsonl(results / "answers.jsonl"), args.example)
        print(f"example question: {example}")
    if "pipeline" in wanted:
        figure_pipeline(results, out, Path(args.data), Path(args.questions), example)
    if "book-map" in wanted:
        figure_book_map(results, out, Path(args.questions), example)
    if "grid" in wanted:
        figure_grid(results, out)
    if "accuracy" in wanted:
        figure_accuracy(results, out)
    if "threshold" in wanted:
        figure_threshold(results, out)


if __name__ == "__main__":
    main()
