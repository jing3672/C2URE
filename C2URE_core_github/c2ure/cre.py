"""CRE: client-local causal response extraction and stable reconstruction."""

from typing import Sequence

from local_causal_energy import (
    client_local_causal_profile,
    train_client_with_invariant_causal_gradients,
)

from .protocol import CREBatch, ClientUpload


class CREModule:
    """Execute prescribed interventions inside each simulated client.

    Raw ``x/adj/y`` are read only in this module (or in a client-local repair
    callback invoked by SRF).  The returned upload contains only the trained
    model/update, sample count, client id, and fixed-length response r_i.
    """

    def collect(self, reference_model, client_models: Sequence,
                client_runtimes: Sequence, *, trials=4, budget=0.2, seed=0,
                device="cpu", outcome="logits"):
        if len(client_models) != len(client_runtimes):
            raise ValueError("client model/runtime counts differ")
        uploads = []
        for index, (client, model) in enumerate(
                zip(client_runtimes, client_models)):
            client_id = int(getattr(client, "cid", index))
            response = client_local_causal_profile(
                reference_model, model, client.x, client.adj,
                trials=trials, budget=budget,
                seed=int(seed) + client_id * 1009,
                device=device, outcome=outcome).detach().cpu().numpy()
            train_idx = getattr(client, "train_idx", ())
            uploads.append(ClientUpload(
                client_id=client_id,
                sample_count=max(int(len(train_idx)), 1),
                model=model,
                causal_response_trials=response,
            ))
        return CREBatch(
            uploads=tuple(uploads),
            trials=max(int(trials), 2),
            budget=float(budget),
            seed=int(seed),
            outcome=str(outcome),
        )

    @staticmethod
    def profile(*args, **kwargs):
        """Compatibility callback for V40's additional local CRE probes."""
        return client_local_causal_profile(*args, **kwargs)

    @staticmethod
    def reconstruct(*args, **kwargs):
        """LCCT-triggered client-local stable-gradient reconstruction."""
        return train_client_with_invariant_causal_gradients(*args, **kwargs)

