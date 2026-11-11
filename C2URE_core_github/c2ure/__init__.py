"""CRE, LCCT and SRF interfaces for the C2URE core implementation."""
from .cre import CREModule
from .lcct import LCCTModule
from .pipeline import C2UREV40Pipeline
from .protocol import CREBatch, ClientUpload, LCCTResult, ModuleTrace
from .srf import SRFModule


def make_pipeline():
    """Create a pipeline around the extracted experiment core."""
    import fl_node
    return C2UREV40Pipeline(fl_node, fl_node.local_causal_energy_aggregate)


__all__ = ["make_pipeline", "CREModule", "LCCTModule", "SRFModule",
           "CREBatch", "ClientUpload", "LCCTResult", "ModuleTrace"]
