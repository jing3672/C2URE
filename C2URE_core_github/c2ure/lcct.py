"""LCCT: server-side scoring using only fixed-length CRE reports."""

import numpy as np

from .protocol import CREBatch, LCCTResult


def _robust_profile_scores(profiles):
    """Exact V40 median/MAD standardization and RMS suspicion score."""
    profiles = np.asarray(profiles, dtype=np.float64)
    center = np.median(profiles, axis=0, keepdims=True)
    mad = np.median(np.abs(profiles - center), axis=0, keepdims=True)
    magnitude = np.median(np.abs(center), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-3 * np.maximum(magnitude, 1e-6))
    standardized = (profiles - center) / scale
    return np.sqrt(np.mean(standardized ** 2, axis=1)), standardized


def _rank_reliability(left, right):
    left_rank = np.argsort(np.argsort(left, kind="mergesort"),
                           kind="mergesort")
    right_rank = np.argsort(np.argsort(right, kind="mergesort"),
                            kind="mergesort")
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


class LCCTModule:
    """Rank causal suspicion without access to client graphs or labels."""

    @staticmethod
    def score(batch: CREBatch):
        tensor = batch.response_tensor
        profiles = tensor.mean(axis=1)
        scores, standardized = _robust_profile_scores(profiles)
        half = max(tensor.shape[1] // 2, 1)
        left_scores, _ = _robust_profile_scores(
            tensor[:, :half].mean(axis=1))
        right_scores, _ = _robust_profile_scores(
            tensor[:, half:].mean(axis=1))
        reliability = _rank_reliability(left_scores, right_scores)
        ranked = tuple(int(value) for value in np.argsort(scores)[::-1])
        return LCCTResult(
            scores=scores,
            profiles=profiles,
            standardized_profiles=standardized,
            reliability=reliability,
            ranked_clients=ranked,
        )

