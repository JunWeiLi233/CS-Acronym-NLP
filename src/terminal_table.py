"""Plain-text tables and dataset selection for terminal commands."""

import shutil
from itertools import zip_longest
from textwrap import wrap as wrap_text


def format_table(headers, rows, *, title=None, right=(), wrap=None):
    """Render a table, optionally wrapping a text column to fit the terminal."""
    # Split cells before sizing so a newline never breaks the table alignment.
    cell_rows = [tuple(str(cell).splitlines() or [""] for cell in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in cell_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], max(map(len, cell)))

    if wrap is not None:
        terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        other_width = sum(widths) - widths[wrap] + 3 * (len(headers) - 1)
        widths[wrap] = min(widths[wrap], max(len(headers[wrap]), terminal_width - other_width))

    def format_line(cells):
        return " | ".join(
            cell.rjust(width) if index in right else cell.ljust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        ).rstrip()

    lines = [title] if title else []
    lines.append(format_line(headers))
    lines.append("-+-".join("-" * width for width in widths))
    for row in cell_rows:
        if wrap is not None:
            wrapped_lines = []
            for line in row[wrap]:
                parts = wrap_text(line, widths[wrap], break_on_hyphens=False)
                wrapped_lines.extend(parts or [""])
            row = list(row)
            row[wrap] = wrapped_lines
        lines.extend(format_line(cells) for cells in zip_longest(*row, fillvalue=""))
    return "\n".join(lines)


def choose_dataset(directory, suffixes=(".json", ".jsonl"), *, title="Available datasets"):
    """List matching datasets and return the selected path, or None to go back."""
    files = sorted(
        file for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() in suffixes
    )
    if not files:
        print(f"No datasets found in {directory}")
        return None

    rows = [(i, file.relative_to(directory)) for i, file in enumerate(files, start=1)]
    rows.append((0, "Back"))
    print("\n" + format_table(("Option", "Dataset"), rows, title=title, right=(0,), wrap=1))

    choice = input("\nSelect a dataset: ").strip()
    if choice == "0":
        return None
    try:
        index = int(choice) - 1
    except ValueError:
        print("Please enter a number.")
        return None
    if not 0 <= index < len(files):
        print("Invalid selection.")
        return None
    return files[index]
