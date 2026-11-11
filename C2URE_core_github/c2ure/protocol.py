"""Typed messages exchanged by the modular C2URE V40 simulation.

The project remains a single-process simulator.  These objects make the
logical communication boundary explicit: LCCT sees uploaded models and a
fixed-length causal response, while raw client graphs remain in the private
runtime list used by CRE and client-local SRF reconstruction.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


@dataclass(frozen=True)
class ClientUpload:
    """One logical client-to-server upload.

    ``model`` represents the uploaded model/update in the existing simulator.
    ``causal_response_trials`` is the fixed-size CRE response; it never holds
    node features, adjacency, labels, train indices, or trigger identities.
    """

    client_id: int
    sample_count: int
    model: Any = field(repr=False)
    causal_response_trials: np.ndarray = field(repr=False)

    def __post_init__(self):
        # Preserve V40's original float32 response dtype.  LCCT performs its
        # own float64 conversion only inside robust standardization; casting
        # here would change the trial mean by a few ulps before scoring.
        response = np.asarray(self.causal_response_trials)
        if response.ndim != 2:
            raise ValueError(
                "causal_response_trials must have shape [trials, features]")
        object.__setattr__(self, "causal_response_trials", response)

    @property
    def causal_response(self):
        """Mean fixed-length response r_i used by LCCT."""
        return self.causal_response_trials.mean(axis=0)


@dataclass(frozen=True)
class CREBatch:
    """CRE output for one federated round."""

    uploads: Tuple[ClientUpload, ...]
    trials: int
    budget: float
    seed: int
    outcome: str

    @property
    def client_models(self):
        return [upload.model for upload in self.uploads]

    @property
    def response_tensor(self):
        if not self.uploads:
            raise ValueError("CRE batch must contain at least one upload")
        return np.stack([
            upload.causal_response_trials for upload in self.uploads
        ])


@dataclass(frozen=True)
class LCCTResult:
    """Server-side causal suspicion evidence produced from CRE reports."""

    scores: np.ndarray
    profiles: np.ndarray
    standardized_profiles: np.ndarray
    reliability: float
    ranked_clients: Tuple[int, ...]

    def as_legacy_tuple(self):
        """Return the exact tuple expected by the preserved V40 core."""
        return (
            self.scores,
            self.profiles,
            self.standardized_profiles,
            self.reliability,
        )


@dataclass(frozen=True)
class ModuleTrace:
    """Non-invasive diagnostic trace; it is not written into V40 state."""

    uploaded_client_ids: Tuple[int, ...]
    ranked_client_indices: Tuple[int, ...]
    reliability: float
    response_dimension: int
    raw_graph_visible_to_lcct: bool = False
