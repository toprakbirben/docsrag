import pytest

from docsrag import config
from docsrag.chunk import chunk, chunk_markdown, chunk_python, window


def body_tokens(c: str) -> int:
    # Markdown chunks are "<breadcrumb>\n\n<body>"; the size limit applies to the body.
    return len(c.split("\n\n", 1)[-1].split())


def test_window_overlap_carries_context_between_neighbours():
    # Overlap exists so a fact straddling a boundary is whole in at least one chunk.
    text = " ".join(f"w{i}" for i in range(11))
    parts = window(text, max_tokens=4, overlap=1)
    assert [p.split() for p in parts] == [
        ["w0", "w1", "w2", "w3"], ["w3", "w4", "w5", "w6"], ["w6", "w7", "w8", "w9"], ["w9", "w10"],
    ]


def test_window_preserves_newlines_so_code_stays_readable():
    assert window("def f():\n    return 1\n", 50, 5) == ["def f():\n    return 1"]


def test_window_rejects_overlap_not_smaller_than_max():
    with pytest.raises(ValueError):
        window("a b c", 2, 2)


def test_markdown_splits_on_headings_and_keeps_breadcrumb():
    md = "# Dependencies\n\nIntro.\n\n## With yield\n\nUse yield for cleanup.\n"
    chunks = chunk_markdown(md, 100, 10)
    assert chunks == ["Dependencies\n\nIntro.", "Dependencies > With yield\n\nUse yield for cleanup."]


def test_markdown_ignores_hash_lines_inside_code_fences():
    # A Python comment in a fenced block must not start a new section.
    md = "# Title\n\n```python\n# not a heading\nx = 1\n```\n"
    assert len(chunk_markdown(md, 100, 10)) == 1


def test_markdown_oversized_section_respects_limit():
    md = "# Big\n\n" + " ".join(["word"] * 1000)
    chunks = chunk_markdown(md, 256, 32)
    assert len(chunks) > 1
    assert all(body_tokens(c) <= 256 for c in chunks)
    assert all(c.startswith("Big\n\n") for c in chunks)


def test_python_function_chunk_includes_its_decorator():
    # Decorators like @app.get("/items") carry the route, which is the answer to many questions.
    src = 'import x\n\n\n@app.get("/items")\ndef read_items():\n    return []\n'
    chunks = chunk_python(src, 100, 10)
    assert 'import x' in chunks[0]
    assert any(c.startswith('@app.get("/items")\ndef read_items') for c in chunks)


def test_python_syntax_error_falls_back_to_window():
    assert chunk_python("def broken(:\n    pass", 100, 10) == ["def broken(:\n    pass"]


def test_python_oversized_class_is_windowed():
    src = "class Big:\n" + "".join(f"    a{i} = {i}\n" for i in range(400))
    assert all(len(c.split()) <= 128 for c in chunk_python(src, 128, 16))


def test_chunk_dispatches_by_extension():
    assert chunk("a.md", "# H\n\nbody", 50, 5) == ["H\n\nbody"]
    assert chunk("a.py", "x = 1", 50, 5) == ["x = 1"]


def test_presets_differ_only_in_what_their_name_says():
    base, hr = config.get("baseline"), config.get("hybrid_rerank")
    assert (hr.hybrid, hr.rerank) == (True, True)
    assert hr.chunker == base.chunker == "w512o64"
    with pytest.raises(ValueError, match="unknown config"):
        config.get("nope")
