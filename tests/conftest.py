"""Safety nets shared by every test module."""

from __future__ import annotations

import contextlib
import os

import pytest

# This environment sets TERM=dumb, NO_COLOR, and FORCE_COLOR=0. Rich then
# ignores an explicit width (80 columns, so assertion strings wrap) and drops
# colour even when a test asked for it. Live updates also skip a dumb terminal,
# so spinners never erase. Tests set COLUMNS and treat the terminal as capable.
# A console that was given a width still uses that width.
os.environ["COLUMNS"] = "200"
os.environ.pop("NO_COLOR", None)
os.environ.pop("FORCE_COLOR", None)


def pytest_configure() -> None:
    import rich.console as rich_console

    original_property = rich_console.Console.size
    original = original_property.fget

    def size(self):
        # Rich returns 80x25 for a dumb terminal as soon as height is unset,
        # even when width was passed in. Honor an explicit width. When width
        # was left unset, use COLUMNS or 200.
        if self.is_dumb_terminal and not (self._width is not None and self._height is not None):
            if self._width is not None:
                width = self._width
            else:
                columns = os.environ.get("COLUMNS", "")
                width = int(columns) if columns.isdigit() else 200
            if self._height is not None:
                height = self._height
            else:
                lines = os.environ.get("LINES", "")
                height = int(lines) if lines.isdigit() else 25
            return rich_console.ConsoleDimensions(width, height)
        return original(self)

    def not_dumb(self) -> bool:
        return False

    rich_console.Console.size = property(size, original_property.fset)
    rich_console.Console.is_dumb_terminal = property(not_dumb)

# The biggest file any test leaves behind today is well under 1 MB.
MAX_TMP_PATH_BYTES = 64 * 1024**2


@pytest.fixture(autouse=True)
def _no_huge_files_in_tmp_path(request):
    """Fail a test whose tmp_path ends up holding more than MAX_TMP_PATH_BYTES.

    A test that needs a "40 GB model" must fake the size, never create the file:
    even ``truncate()`` is only free on file systems with sparse files. NTFS
    really allocates it, which once filled a Windows CI runner's disk and broke
    hundreds of unrelated tests. Sizes are the files' apparent sizes, so a
    sparse file that costs nothing on Linux is caught here too; the offending
    files are deleted, so the rest of the run keeps its disk space.
    """
    if "tmp_path" not in request.fixturenames:  # (also true for tmpdir and fixtures built on tmp_path)
        yield
        return
    root = request.getfixturevalue("tmp_path")
    yield
    sizes = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                sizes.append((os.lstat(path).st_size, path))
            except OSError:
                continue
    total = sum(size for size, _path in sizes)
    if total > MAX_TMP_PATH_BYTES:
        biggest = ", ".join(f"{os.path.relpath(p, root)} ({size / 1024**2:,.0f} MB)"
                            for size, p in sorted(sizes, reverse=True)[:3])
        for size, path in sizes:
            if size >= 1024**2:  # don't let one test starve the rest of the run of disk space
                with contextlib.suppress(OSError):
                    os.remove(path)
        pytest.fail(
            f"This test left {total / 1024**2:,.0f} MB in its tmp_path (limit {MAX_TMP_PATH_BYTES // 1024**2} MB): "
            f"{biggest} (now deleted). Fake big file sizes instead of creating big files - on Windows (NTFS) "
            "even a truncate()d file really takes up that much disk.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _never_open_a_real_browser(monkeypatch):
    """Tests must never launch the machine's web browser (one once left Chromium
    running on a CI runner). Anything that reaches the real ``webbrowser`` is
    recorded here instead; tests that care inject their own opener."""
    import webbrowser

    opened: list[str] = []

    def record(url, *args, **kwargs):
        opened.append(url)
        return True

    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(webbrowser, name, record)
    return opened

