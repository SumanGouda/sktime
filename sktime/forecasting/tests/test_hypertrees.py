
"""Regression tests for HyperTree forecasters."""

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.hypertrees import (
    HyperTreeARForecaster,
    HyperTreeNetARForecaster,
)


FORECASTERS = [HyperTreeARForecaster, HyperTreeNetARForecaster]


@pytest.mark.parametrize("forecaster_class", FORECASTERS)
@pytest.mark.skipif(
    not all(
        _check_estimator_deps(est, severity="none")
        for est in FORECASTERS
    ),
    reason="hypertrees-forecasting dependency is not installed",
)
def test_month_end_frequency_pandas3(forecaster_class, monkeypatch):
    """Test that the legacy month-end alias is normalized for pandas 3."""
    index = pd.date_range("2000-01-31", periods=12, freq="ME")
    y = pd.Series(
        range(12),
        index=pd.DatetimeIndex(index.to_numpy()),
        dtype=float,
    )

    # Simulate legacy frequency inference.
    monkeypatch.setattr(pd, "infer_freq", lambda index: "M")

    if forecaster_class is HyperTreeARForecaster:
        from hypertrees.models.HyperTreeAR import HyperTreeAR

        monkeypatch.setattr(HyperTreeAR, "train", lambda self, *args, **kwargs: None)
    else:
        from hypertrees.models.HyperTreeNetAR import HyperTreeNetAR

        monkeypatch.setattr(
            HyperTreeNetAR, "train", lambda self, *args, **kwargs: None
        )

    forecaster = forecaster_class()
    forecaster.fit(y, fh=[1])

    assert forecaster._freq == "ME"
    assert forecaster._dates.freqstr == "ME"
