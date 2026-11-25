from __future__ import annotations

import pytest

from rag_local.documents import (
    find_chapters,
    gutenberg_span,
    load_book,
    load_books,
    roman_to_int,
    split_book,
)

from .conftest import BOOK


def test_roman_numerals():
    assert [roman_to_int(n) for n in ["I", "IV", "IX", "XII", "XIV", "XL", "XC"]] == [
        1,
        4,
        9,
        12,
        14,
        40,
        90,
    ]


def test_header_and_licence_are_cut(books_dir):
    book = load_book(books_dir / "test_book.md")
    assert book.body.startswith("Contents")
    assert book.body.endswith("somersault.")
    assert "Project Gutenberg" not in book.body
    assert "licence" not in book.body
    assert "\r" not in book.text


def test_markdown_style_start_marker():
    text = (
        "header\n**_ START OF THE PROJECT GUTENBERG EBOOK X _**\nBody text.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence"
    )
    start, end = gutenberg_span(text)
    assert text[start:end] == "Body text."


def test_text_without_markers_is_kept_whole():
    assert gutenberg_span("  just a story  ") == (2, 14)


def test_keep_boilerplate(books_dir):
    book = load_book(books_dir / "test_book.md", strip_boilerplate=False)
    assert (book.start, book.end) == (0, len(book.text))
    assert "General terms of use" in book.body


def test_chapter_headings_skip_the_contents(books_dir):
    book = load_book(books_dir / "test_book.md")
    assert [number for _, number in book.chapter_starts] == [1, 2]
    contents = book.text.index("CHAPTER I. The Garden")
    assert book.chapter_at(contents) == 0
    assert book.chapter_at(book.text.index("busily painting")) == 1
    assert book.chapter_at(book.text.index("A whiting asked")) == 2


@pytest.mark.parametrize("size,overlap", [(120, 20), (200, 40), (1000, 200)])
def test_chunk_offsets_point_into_the_file(books_dir, size, overlap):
    book = load_book(books_dir / "test_book.md")
    chunks = split_book(book, size, overlap)
    assert len(chunks) > 1 or size == 1000
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert book.text[chunk.start : chunk.end] == chunk.text
        assert book.start <= chunk.start < chunk.end <= book.end
        assert len(chunk.text) <= size
        assert chunk.id == f"test_book:{chunk.start}-{chunk.end}"
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_chunk_chapters(books_dir):
    chunks = split_book(load_book(books_dir / "test_book.md"), 200, 40)
    garden = [c.chapter for c in chunks if "gardeners" in c.text]
    sea = [c.chapter for c in chunks if "lobsters" in c.text]
    assert set(garden) == {1}
    assert set(sea) == {2}


def test_load_books_needs_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_books(tmp_path)


def test_the_real_book():
    book = load_book(BOOK)
    assert book.path.endswith("alice_in_wonderland.md")
    assert not book.path.startswith("/")
    assert len(book.chapter_starts) == 12
    assert [n for _, n in book.chapter_starts] == list(range(1, 13))
    assert book.body.startswith("[Illustration]")
    assert book.body.endswith("THE END")
    assert "Project Gutenberg" not in book.body
    # The header and licence make up about a tenth of the file.
    assert 0.08 < 1 - len(book.body) / len(book.text) < 0.15
    assert find_chapters(book.text) == book.chapter_starts
