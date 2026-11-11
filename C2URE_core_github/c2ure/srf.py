"""SRF/CRFU: server-orchestrated hybrid repair and fusion."""

import threading


class SRFModule:
    """Route LCCT evidence and CRE repair callbacks into frozen V40 logic.

    V40 remains a single-process simulator.  The preserved core still owns
    Safe Core, utility-controlled release, Shadow Guard, and Server Purify.
    Client graph objects are supplied only as the private runtime needed by
    local reconstruction and the existing Non-IID policy calculation.
    """

    def __init__(self, core_module, preserved_aggregate, cre_module):
        self.core = core_module
        self.preserved_aggregate = preserved_aggregate
        self.cre = cre_module
        self._lock = threading.RLock()

    def fuse(self, reference_model, cre_batch, client_runtimes, state,
             lcct_result, *args, **kwargs):
        """Run unchanged V40 repair/fusion with precomputed CRE/LCCT output."""
        if len(cre_batch.uploads) != len(client_runtimes):
            raise ValueError("CRE upload/runtime counts differ")

        def precomputed_scores(*_args, **_kwargs):
            return lcct_result.as_legacy_tuple()

        # The preserved implementation imports these functions into module
        # globals.  Patch only during this sequential simulator call and
        # restore unconditionally.  No source file or persistent V40 state is
        # changed by the modular boundary.
        with self._lock:
            original_scores = self.core.compute_local_causal_scores
            original_profile = self.core.client_local_causal_profile
            original_reconstruct = (
                self.core.train_client_with_invariant_causal_gradients)
            try:
                self.core.compute_local_causal_scores = precomputed_scores
                self.core.client_local_causal_profile = self.cre.profile
                self.core.train_client_with_invariant_causal_gradients = (
                    self.cre.reconstruct)
                return self.preserved_aggregate(
                    reference_model,
                    cre_batch.client_models,
                    client_runtimes,
                    state,
                    *args,
                    **kwargs,
                )
            finally:
                self.core.compute_local_causal_scores = original_scores
                self.core.client_local_causal_profile = original_profile
                self.core.train_client_with_invariant_causal_gradients = (
                    original_reconstruct)

