"""CRE -> LCCT -> SRF orchestration for the preserved C2URE V40 method."""

import functools
import inspect

from .cre import CREModule
from .lcct import LCCTModule
from .protocol import ModuleTrace
from .srf import SRFModule


class C2UREV40Pipeline:
    """Behavior-preserving modular facade over the frozen V40 aggregate."""

    def __init__(self, core_module, preserved_aggregate):
        self.core = core_module
        self.preserved_aggregate = preserved_aggregate
        self.cre = CREModule()
        self.lcct = LCCTModule()
        self.srf = SRFModule(core_module, preserved_aggregate, self.cre)
        self.signature = inspect.signature(preserved_aggregate)
        self.last_trace = None

    def run_round(self, reference_model, models, clients, state,
                  *args, **kwargs):
        bound = self.signature.bind_partial(
            reference_model, models, clients, state, *args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        batch = self.cre.collect(
            reference_model,
            models,
            clients,
            trials=options["trials"],
            budget=options["budget"],
            seed=options["seed"],
            device=options["device"],
            outcome=options["outcome"],
        )
        lcct_result = self.lcct.score(batch)
        model = self.srf.fuse(
            reference_model,
            batch,
            clients,
            state,
            lcct_result,
            *args,
            **kwargs,
        )
        self.last_trace = ModuleTrace(
            uploaded_client_ids=tuple(
                upload.client_id for upload in batch.uploads),
            ranked_client_indices=lcct_result.ranked_clients,
            reliability=float(lcct_result.reliability),
            response_dimension=int(lcct_result.profiles.shape[1]),
        )
        return model


def install_into(core_module):
    """Install the modular facade into an in-memory V40 module only."""
    existing = getattr(core_module, "_c2ure_v40_modular_pipeline", None)
    if existing is not None:
        return existing
    preserved = core_module.local_causal_energy_aggregate
    pipeline = C2UREV40Pipeline(core_module, preserved)

    @functools.wraps(preserved)
    def modular_aggregate(reference_model, models, clients, state,
                          *args, **kwargs):
        return pipeline.run_round(
            reference_model, models, clients, state, *args, **kwargs)

    modular_aggregate.__signature__ = inspect.signature(preserved)
    core_module._c2ure_v40_preserved_aggregate = preserved
    core_module._c2ure_v40_modular_pipeline = pipeline
    core_module.local_causal_energy_aggregate = modular_aggregate
    return pipeline


def uninstall_from(core_module):
    """Restore the in-memory module; source files are never modified."""
    preserved = getattr(
        core_module, "_c2ure_v40_preserved_aggregate", None)
    if preserved is not None:
        core_module.local_causal_energy_aggregate = preserved
    if hasattr(core_module, "_c2ure_v40_modular_pipeline"):
        delattr(core_module, "_c2ure_v40_modular_pipeline")
    if hasattr(core_module, "_c2ure_v40_preserved_aggregate"):
        delattr(core_module, "_c2ure_v40_preserved_aggregate")

