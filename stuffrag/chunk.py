import ast
import re

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
TOKEN = re.compile(r"\S+\s*")  # word plus trailing whitespace, so joins keep newlines


def window(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split into windows of <= max_tokens words; neighbours share `overlap` words."""
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"need 0 <= overlap < max_tokens, got {overlap=} {max_tokens=}")
    toks = TOKEN.findall(text)
    if not toks:
        return []
    if len(toks) <= max_tokens:
        return ["".join(toks).strip()]
    step = max_tokens - overlap
    return ["".join(toks[i : i + max_tokens]).strip() for i in range(0, len(toks) - overlap, step)]


def chunk_markdown(text: str, max_tokens: int, overlap: int) -> list[str]:
    """One section per heading, prefixed with its heading path; big sections are windowed."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((" > ".join(path), body))
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            path = path[: len(m.group(1)) - 1] + [m.group(2).strip()]
        else:
            lines.append(line)
    flush()

    return [
        f"{crumb}\n\n{piece}" if crumb else piece
        for crumb, body in sections
        for piece in window(body, max_tokens, overlap)
    ]


def chunk_python(source: str, max_tokens: int, overlap: int) -> list[str]:
    """One chunk per top-level function/class (with decorators) plus one for module-level code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return window(source, max_tokens, overlap)
    lines = source.splitlines()
    used: set[int] = set()
    defs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            used.update(range(start, node.end_lineno))
            defs.extend(window("\n".join(lines[start : node.end_lineno]), max_tokens, overlap))
    rest = "\n".join(line for i, line in enumerate(lines) if i not in used)
    return window(rest, max_tokens, overlap) + defs


def chunk(path: str, text: str, max_tokens: int, overlap: int) -> list[str]:
    if path.endswith(".md"):
        return chunk_markdown(text, max_tokens, overlap)
    if path.endswith(".py"):
        return chunk_python(text, max_tokens, overlap)
    return window(text, max_tokens, overlap)
