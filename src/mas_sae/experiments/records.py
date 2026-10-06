from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write experiment records as JSON Lines."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_jsonl(
    path: str | Path,
    row: dict[str, Any],
) -> None:
    """Append one record durably (flush and fsync) so restarts can trust it."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Read JSON Lines experiment records."""

    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]
