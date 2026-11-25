"""Load books, cut away the Project Gutenberg header and licence, and split the text into chunks.

Every chunk keeps its character offsets in the book file (after line endings are normalised to
"\\n"), so ``read_text(path)[chunk.start:chunk.end] == chunk.text`` always holds.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

_START = re.compile(r"^[^\n]*\bSTART OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b[^\n]*$", re.M | re.I)
_END = re.compile(r"^[^\n]*\bEND OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b[^\n]*$", re.M | re.I)
# A heading alone on its line; the table of contents has the title on the same line and is skipped.
_CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)\.[ \t]*$", re.M)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(numeral: str) -> int:
    total, largest = 0, 0
    for char in reversed(numeral.upper()):
        value = _ROMAN[char]
        if value < largest:
            total -= value
        else:
            total += value
            largest = value
    return total


def read_text(path: str | Path) -> str:
    """Read a UTF-8 file; universal newlines turn CRLF into LF and a BOM is dropped."""
    with open(path, encoding="utf-8-sig") as handle:
        return handle.read()


def display_path(path: str | Path) -> str:
    """The path relative to the working directory, or just the file name if it lies elsewhere."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def gutenberg_span(text: str) -> tuple[int, int]:
    """Character span of the book proper, between the Project Gutenberg start and end markers.

    Text without markers is returned whole. Surrounding whitespace is trimmed.
    """
    start, end = 0, len(text)
    match = _START.search(text)
    if match:
        start = match.end()
    match = _END.search(text, start)
    if match:
        end = match.start()
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


@dataclass(frozen=True)
class Book:
    path: str
    text: str
    start: int
    end: int
    sha256: str
    chapter_starts: tuple[tuple[int, int], ...] = field(default=(), repr=False)

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def body(self) -> str:
        return self.text[self.start : self.end]

    def chapter_at(self, offset: int) -> int:
        """The chapter number that contains ``offset``; 0 before the first chapter heading."""
        positions = [pos for pos, _ in self.chapter_starts]
        index = bisect.bisect_right(positions, offset) - 1
        return self.chapter_starts[index][1] if index >= 0 else 0


def find_chapters(text: str, start: int = 0, end: int | None = None) -> tuple[tuple[int, int], ...]:
    """(offset, number) for each 'CHAPTER I.' style heading between ``start`` and ``end``."""
    end = len(text) if end is None else end
    return tuple(
        (match.start(), roman_to_int(match.group(1)))
        for match in _CHAPTER.finditer(text, start, end)
    )


def load_book(path: str | Path, strip_boilerplate: bool = True) -> Book:
    text = read_text(path)
    start, end = gutenberg_span(text) if strip_boilerplate else (0, len(text))
    return Book(
        path=display_path(path),
        text=text,
        start=start,
        end=end,
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        chapter_starts=find_chapters(text, start, end),
    )


def load_books(
    data_dir: str | Path, pattern: str = "*.md", strip_boilerplate: bool = True
) -> list[Book]:
    paths = sorted(Path(data_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {display_path(data_dir)}")
    return [load_book(path, strip_boilerplate) for path in paths]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    start: int
    end: int
    chapter: int

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "chapter": self.chapter,
        }


def split_book(book: Book, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split the book with LangChain's recursive splitter, keeping offsets into the book file.

    A chunk's chapter is the one that holds its middle character.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    body = book.body
    chunks = []
    for doc in splitter.create_documents([body]):
        local = doc.metadata["start_index"]
        if local < 0 or body[local : local + len(doc.page_content)] != doc.page_content:
            raise RuntimeError(f"could not place a chunk of {book.path} in the text")
        start = book.start + local
        end = start + len(doc.page_content)
        chunks.append(
            Chunk(
                id=f"{book.name}:{start}-{end}",
                text=doc.page_content,
                source=book.path,
                start=start,
                end=end,
                chapter=book.chapter_at((start + end) // 2),
            )
        )
    return chunks


def split_books(books: list[Book], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return [chunk for book in books for chunk in split_book(book, chunk_size, chunk_overlap)]
