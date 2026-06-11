from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Any


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames) if fieldnames else None
        self._file = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = None

    def log(self, row: Mapping[str, Any]) -> None:
        row = dict(row)
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction='ignore')
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
