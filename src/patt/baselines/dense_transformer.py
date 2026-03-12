from __future__ import annotations

from ..models.patt import PATT


class DenseTransformerForecaster(PATT):
    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["attention_mode"] = "dense"
        super().__init__(**kwargs)
