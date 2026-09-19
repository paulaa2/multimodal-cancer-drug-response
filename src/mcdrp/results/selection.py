"""Row selection shared by the reporting scripts.

Every report needs "the best model for this split", and the obvious spelling
of that, ``sort_values("rmse").groupby(keys).first()``, is wrong here. pandas
reduces each column independently and skips NaN, so the resulting row is a
blend: it takes the winner's RMSE but fills each hyperparameter column from
whichever *other* model first had a value there. Stages report disjoint
hyperparameter columns, so the blend happens on essentially every row rather
than in some edge case.

Keeping the correct spelling in one place stops the wrong one from being
reintroduced in the next report script.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def lowest_rmse_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    rmse_column: str = "rmse",
) -> pd.DataFrame:
    """Return the intact row with the lowest RMSE within each key group.

    Ties keep the first row in the frame's existing order, so callers can make
    the choice deterministic by sorting beforehand.
    """

    keys = list(keys)
    missing = [key for key in [*keys, rmse_column] if key not in frame.columns]
    if missing:
        raise KeyError(f"Selection columns absent from the frame: {missing}")

    ordered = frame.sort_values([*keys, rmse_column], kind="stable")
    return ordered.groupby(keys, as_index=False, dropna=False).head(1)
