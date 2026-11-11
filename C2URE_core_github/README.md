# C2URE

Core implementation of **C²URE: Backdoor-Resilient Federated Graph Learning through Intervention and Repair**.

## Method

- **CRE** measures client-update responses under randomized structural, feature, and joint interventions.
- **LCCT** ranks response deviations and audits their reproducibility across intervention trials.
- **SRF** reconstructs and repairs candidate contributions, controls their release, and aggregates updates.

The simulator assumes prescribed response generation and repair. Trusted server data support task calibration and independent utility monitoring.

## Files

| File | Content |
|---|---|
| `c2ure/cre.py` | Causal response extraction and reconstruction interface |
| `c2ure/lcct.py` | Robust response scoring and reliability audit |
| `c2ure/srf.py` | Repair/fusion orchestration |
| `c2ure/pipeline.py` | CRE → LCCT → SRF pipeline |
| `c2ure/protocol.py` | Intermediate response and tracing structures |
| `fl_node.py` | Core aggregation, repair, temporal routing, trusted utility checks, and client training |
| `local_causal_energy.py` | Graph interventions, response statistics, and stable-gradient reconstruction |
| `models_node.py` | Node GCN and GraphSAGE model definitions |

This repository contains the core algorithm. Dataset loaders, attack implementations, experiment queues, historical backups, and the full benchmark suite are outside this release.

## Installation and integration

Use Python 3.9 or later and install a PyTorch build appropriate for your hardware:

```bash
python -m pip install -r requirements.txt
```

Create a pipeline and pass your reference model, locally trained client models, client runtime objects, and persistent server state:

```python
from c2ure import make_pipeline

pipeline = make_pipeline()
server_state = {}

# Client runtimes expose cid, x, adj, y, and train_idx.
# reference_model and client_models follow the interface in models_node.py.
repaired_model = pipeline.run_round(
    reference_model,
    client_models,
    client_runtimes,
    server_state,
    trials=4,
    budget=0.2,
    keep_ratio=0.7,
    repair_mode="invariant_shadow_guarded",
    seed=10,
    device="cpu",
)
```

`fl_node.local_causal_energy_aggregate` exposes the complete core parameter signature, including separate `release_calibration_probe` and `release_monitor_probe` inputs. Integration options above illustrate the API; reproducing the paper's full configuration requires the corresponding training and evaluation setup.

The underlying implementation also exposes `train_trust_recovery_local` and `crossfit_server_utility_repair` for the final-model safeguards.

## Provenance

Core functions were extracted from the experiment implementation with their function bodies and decorators unchanged. Supporting modules and the five existing interface modules retain their original contents. `source_manifest.json` records the extracted function names and checksums. The packaging adds the public `make_pipeline()` entry point.
