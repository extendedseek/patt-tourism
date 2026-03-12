from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = []

    def log(self, **kwargs) -> None:
        self.rows.append(kwargs)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
