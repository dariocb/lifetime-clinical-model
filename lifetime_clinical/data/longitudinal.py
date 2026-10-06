"""Independent patient GP blocks with forecast-origin context and irregular rows.

The prior encoder sees only information before the block origin. Target values
are GP observations, not learned-kernel inputs. This fixed-context convention
defines a normalized conditional joint likelihood within each block.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .preprocessing import (VARIABLES, prepare_continuous_targets, build_causal_batches)


@dataclass
class GPBlock:
    patient_id: str
    static: np.ndarray
    history: np.ndarray
    ages: np.ndarray  # age/100
    variables: np.ndarray
    y: np.ndarray  # standardized values
    offsets: np.ndarray  # variable-specific train means, length n_variables
    scales: np.ndarray  # variable-specific train standard deviations
    origin: object = None

    def validate(self):
        n = len(self.y)
        if self.static.ndim != 1 or self.history.ndim != 2:
            raise ValueError("Expected static vector and history token matrix")
        if self.ages.shape != (n,) or self.variables.shape != (n,) or self.y.shape != (n,):
            raise ValueError("Inconsistent irregular target rows")
        if self.offsets.shape != self.scales.shape or self.offsets.ndim != 1:
            raise ValueError("Invalid per-variable scaling metadata")
        if (not np.issubdtype(self.variables.dtype, np.integer)
                or np.any(self.variables < 0) or np.any(self.variables >= len(self.scales))
                or not np.equal(self.variables, np.floor(self.variables)).all()):
            raise ValueError("Invalid physiological variable IDs")
        arrays = [self.static, self.history, self.ages, self.y, self.offsets, self.scales]
        if not all(np.isfinite(a).all() for a in arrays) or np.any(self.scales <= 0):
            raise ValueError("Nonfinite inputs or nonpositive scales")
        if np.any(np.diff(self.ages) < 0):
            raise ValueError("Rows must be ordered by age; simultaneous variables are allowed")
        return self

    def subset(self, indices):
        return replace(self, ages=self.ages[indices].copy(), variables=self.variables[indices].copy(),
                       y=self.y[indices].copy()).validate()

    @property
    def raw_y(self):
        return self.offsets[self.variables] + self.scales[self.variables] * self.y


def prepare_gp_blocks(dataset, preprocessor, patient_ids=None, history_fraction=.2):
    """Prefix panels form context; later panels form the GP block.

    The target-origin panel is excluded from context, including same-time events.
    Unknown/missing measurement targets are dropped, not filled. No oracle history
    is used. All rows within a panel share the same age; panels need not align with
    visits. history_fraction controls the prefix count, not calendar duration.
    """
    if not 0 <= history_fraction < 1:
        raise ValueError("history_fraction must be in [0,1)")
    targets = prepare_continuous_targets(dataset, preprocessor, patient_ids)
    blocks = []
    for pid, rows in targets[targets.target_mask].groupby("patient_id", sort=True):
        rows = rows.sort_values(["measurement_date", "variable_id"])
        times = rows.measurement_date.drop_duplicates().tolist()
        origin = times[min(int(len(times) * history_fraction), len(times)-1)]
        targets_at_origin = rows[rows.measurement_date >= origin]
        batch = build_causal_batches(dataset, preprocessor,
                                     pd.DataFrame([dict(patient_id=pid, query_time=origin)]))
        history = batch["history"][0, batch["history_mask"][0]]
        offsets = np.array([preprocessor.measurement_scales[v][0] for v in VARIABLES])
        scales = np.array([preprocessor.measurement_scales[v][1] for v in VARIABLES])
        blocks.append(GPBlock(pid, batch["static"][0].astype(float), history.astype(float),
                              targets_at_origin.query_age_scaled.to_numpy(dtype=float),
                              targets_at_origin.variable_id.to_numpy(dtype=int),
                              targets_at_origin.target_z.to_numpy(dtype=float), offsets, scales, origin).validate())
    return blocks


def split_forecast_block(block, fraction=.6):
    """Hold out complete future panels; never split a simultaneous panel."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    ages = np.unique(block.ages)
    if len(ages) < 2:
        raise ValueError("At least two panels required for a prospective forecast")
    cutoff = ages[min(max(int(len(ages)*fraction), 1), len(ages)-1)]
    return block.subset(block.ages < cutoff), block.subset(block.ages >= cutoff)
