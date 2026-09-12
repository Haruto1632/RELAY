# RELAY v1.2 implementation status

Status: **environment implementation complete** on 2026-09-02.

This status covers the executable environment, deterministic scenario corpus,
MAPPO training/evaluation infrastructure, provenance, replay-only viewer, controlled
P3 fixtures, and release checks specified by the RELAY Environment Technical Design
FINAL v1.2. It does not claim that the long scientific training matrix has been run
or that publication results have been produced.

## Frozen artifacts

| Artifact | Result |
|---|---|
| Python runtime | CPython 3.11.9, pinned by `.python-version` and `pyproject.toml` |
| Dependency lock | `uv.lock`, Windows training pinned to PyTorch CUDA 13.0 |
| Pilot manifest | 8,000 train / 1,000 validation / 1,000 test; checksum `b85d8cbb27bb6bb1ea64e3625d0f5d05731e88098416ff91bbefac6e888f7973` |
| P3 manifest | 8,000 train / 1,000 validation / 1,000 test; checksum `307348ddf8b41aa855c7d8142b9deb62a5b24a6cc0aedef4cb7be9c690470d0c` |
| Calibration | 100 random and 100 scripted-oracle pilot episodes in `output/calibration`; oracle completion 100%, random completion 0% |
| Controlled P3 cases | All nine fixtures in `configs/fixtures/p3_core.yaml` |

Both manifests were exhaustively validated: 20,000 of 20,000 scenarios satisfy the
generator constraints.

## Technical-design acceptance matrix

| ID | Implemented evidence |
|---|---|
| T-DET-01 | Reset/action-log byte determinism and canonical SHA-256 hashes |
| T-DET-02 | Scalar/process-vector parity tests |
| T-GEN-01 | Exhaustive manifest validator plus generator property tests |
| T-OBS-01 | Metamorphic unseen-state leakage test |
| T-OBS-02 | Wall/line-of-sight occlusion test |
| T-ACT-01 | Edge/wall action-mask and execution tests |
| T-ROLE-01 | Wrong-role intervention rejection test |
| T-ROLE-02 | Reservation, service duration, and single-resolution reward test |
| T-COM-01 | All five communication modes tested for SEND/recipient/cost semantics |
| T-COM-02 | Channel ablation physical-trajectory invariance test |
| T-RWD-01 | Ordered float32 reward reconstruction test |
| T-REP-01 | Parquet replay hash verification and log-only renderer test |
| T-API-01 | PettingZoo parallel API compliance and Gymnasium space tests |
| T-CFG-01 | Typed composition, invalid-key rejection, fixed-field locks, checksums |
| T-EXE-01 | Worker-layout invariant seed and trajectory tests; exact checkpoints |
| T-LOG-01 | Atomic run lifecycle and partitioned Parquet logging tests |
| T-VIS-01 | Offscreen GUI seek/control/overlay immutability and export tests |

The suite also checks a fresh-instance 240-tick `pilot_core` replay, complete frozen
manifest split integrity, scripted-oracle feasibility, environment/vector RNG
restoration, and replay provenance.

## Operational completion

- `relay.train` supports strict deterministic execution and exact `--resume` of
  model, optimizer, recurrent state, RNG state, rollout cursor, and scenario cursor.
- CUDA training is verified on an RTX 5070 Laptop GPU with compute capability 12.0;
  activation discovers the CUDA 13.3 toolkit and the lock selects PyTorch `cu130`.
- Training, evaluation, and baseline commands provide live progress, throughput,
  running metrics, ETA, and checkpoint/export notifications.
- `relay.evaluate` consumes ordered frozen manifest splits and writes every full
  trajectory with run/checkpoint/config provenance.
- `relay.replay` summarizes, verifies, generates, views, and exports immutable
  replay artifacts as PNG, GIF, or MP4.
- The replay desktop interface provides transport, random seek, speed, agent/global
  perspectives, overlays, inspection, and deterministic replay generation.
- Scientific comparison fields are locked in the checked-in P1, P2, P3, O1, and V1
  experiment configurations.
- `ci/run_release_checks.py` is the provider-independent quick/full release gate.

The implementation is therefore ready for edits and scientific experiment runs;
the environment itself does not need another build phase.
