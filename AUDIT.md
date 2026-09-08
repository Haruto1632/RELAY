# RELAY implementation audit

Audit date: 2026-09-08  
Audited implementation: `relay-grid-v1` in this repository  
Scope: environment mechanics, spaces, communication, model, MAPPO training,
evaluation, determinism, logging, replay, configurations, and current artifacts.

This document describes what the code actually does. The shorter `README.md` is
the operating guide.

## 1. Executive summary

RELAY is a deterministic, cooperative, heterogeneous multi-agent search-and-rescue
grid environment exposed through the PettingZoo parallel API. A shared recurrent
actor controls every agent from local observations. During training, a centralized
critic receives privileged world state. Communication is an explicit component of
the policy action and has five ablation modes.

The implementation is functional and its quick release gate passes 31 tests plus
Ruff and strict mypy. The environment, training, CUDA execution, evaluation,
checkpointing, Parquet replay verification, desktop viewer, and media export have
all been exercised locally.

Important interpretation points:

- This is not a transport-to-hospital task. An Ambulance resolves a victim in place.
- The central `staging` cells are spawn/map features, not a rescue destination.
- Communication has unlimited spatial range. Walls and distance do not block it.
- Agents do not collide and may occupy or pass through the same free cell.
- All agents receive the same team reward.
- Environment simulation remains CPU-side; neural inference and PPO optimization
  run on the configured PyTorch device.
- Training does not run periodic validation automatically. Use `relay-evaluate`
  against a saved checkpoint.

## 2. Project entry points

| Command | Function |
|---|---|
| `relay-train` | Resolve a YAML config, create/resume a run, train recurrent MAPPO |
| `relay-evaluate` | Run a frozen checkpoint on ordered evaluation seeds and save replays |
| `relay-replay` | Baseline, generate, summarize, verify, view, or export replay data |
| `relay-scenarios` | Generate/validate seed manifests or create calibration data |
| `relay-baseline` | Direct alias for the paired random NoComm/AlwaysComm baseline |

`Activate-Relay.ps1` activates the project-local CPython 3.11.9 environment,
repairs its local runtime path when necessary, discovers the newest installed CUDA
Toolkit, and creates stable PowerShell command functions.

## 3. Presets and entities

| Preset | Grid | Team | Incidents | Horizon | Internal walls | Service |
|---|---:|---|---|---:|---:|---:|
| `smoke_ci` | 11 x 11 | 1 Scout, 1 Ambulance, 1 Fireman | 1 victim, 1 fire | 80 | 10% | 2 ticks |
| `pilot_core` | 17 x 17 | 1 Scout, 1 Ambulance, 1 Fireman | 3 victims, 3 fires | 240 | 14% | 3 ticks |
| `p3_core` | 21 x 21 | 1 Scout, 2 Ambulances, 2 Firemen | 4 victims, 4 fires | 360 | 16% | 3 ticks |

Fixed agent slots are, in order:

1. `scout_0`
2. `ambulance_0`
3. `ambulance_1`
4. `fireman_0`
5. `fireman_1`

The preset selects a subset of those slots. Sender and recipient identities always
use the same five-slot encoding.

### Roles

- Scout: sensor radius 4 and movement distance 2 cells per movement action. It
  cannot resolve either incident type.
- Ambulance: sensor radius 2 and movement distance 1. It can resolve victims only.
- Fireman: sensor radius 2 and movement distance 1. It can resolve fires only.

Sensor radius is a Chebyshev square around the agent, clipped to the map, with
deterministic supercover line-of-sight. A blocking wall is visible; cells behind it
are hidden.

## 4. Environment lifecycle and tick order

`reset(seed=...)` deterministically generates walls, staging, spawn positions,
incidents, a service tie-break order, and isolated channel RNG state. The initial
actor observation and SHA-256 state hash are returned for every agent.

Every `step(joint_action)` requires exactly one valid action dictionary for every
currently active agent. A tick is resolved in this order:

1. Validate the complete joint action without mutating state.
2. Attempt/queue communication packets and charge attempts.
3. Resolve all physical movement from the pre-movement state.
4. Record first discoveries from post-movement positions.
5. Progress existing interventions, then start new valid interventions.
6. Increment the world tick.
7. Deliver packets whose delivery tick is now due.
8. Determine completion or horizon truncation.
9. Construct the shared reward and next observations.
10. Store transition diagnostics and the post-step state hash.

All physical destinations are applied simultaneously. There is no occupancy or
collision constraint, so multiple agents can share a cell. Incidents do not block
movement.

For a Scout's two-cell move, the environment advances one cell at a time and stops
at the first wall. The public action mask checks the immediately adjacent cell;
therefore a masked-legal Scout move can still be shortened by a wall in its second
cell. The `move_blocked_or_shortened` info field records this.

## 5. Action space

Every agent always submits the same Gymnasium `Dict` action:

| Key | Space | Meaning |
|---|---|---|
| `physical` | `Discrete(6)` | Physical action listed below |
| `send` | `Discrete(2)` | 0 = do not send, 1 = send; ignored in some modes |
| `recipient` | `Discrete(5)` | Fixed recipient slot; ignored in broadcast/NoComm modes |
| `message` | `Box(-1, 1, (8,), float32)` | Learned continuous payload |

Physical actions:

| ID | Action | Meaning |
|---:|---|---|
| 0 | `STAY` | Remain in place |
| 1 | `MOVE_N` | Move toward decreasing y |
| 2 | `MOVE_E` | Move toward increasing x |
| 3 | `MOVE_S` | Move toward increasing y |
| 4 | `MOVE_W` | Move toward decreasing x |
| 5 | `INTERACT` | Begin the role-compatible incident service on the current cell |

Illegal physical actions are not rejected after validation; they execute as
`STAY`, with `invalid_masked_action=True`. While busy, only `STAY` is physically
legal, but the agent can still communicate. `INTERACT` is legal only when an
Ambulance is on an active victim or a Fireman is on an active fire. The Scout and
wrong specialist role cannot interact.

When multiple same-role specialists try to reserve the same incident on the same
tick, the seeded scenario tie-break order chooses exactly one winner. The incident
becomes `reserved`; after the configured total service ticks it becomes `resolved`.
No return journey is required.

## 6. Actor observation space

Each actor sees only its local observation and received messages:

| Key | Shape | Contents |
|---|---:|---|
| `local_grid` | `(10, 9, 9)` | North-up local crop described below |
| `self_vec` | `(16,)` | Position, role, busy/action history, normalized time |
| `physical_action_mask` | `(6,)` | Legal physical actions |
| `recipient_mask` | `(5,)` | Present teammates except self |
| `message_payloads` | `(4, 8)` | Payloads delivered on this tick |
| `message_sender_ids` | `(4, 5)` | One-hot fixed sender slot |
| `message_sender_roles` | `(4, 3)` | Scout/Ambulance/Fireman one-hot |
| `message_mask` | `(4,)` | Which message rows are populated |

All observation arrays are read-only. The crop is always 9 x 9 and centered on
the observing agent. Invisible and out-of-map cells are all-zero.

`local_grid` channels:

| Channel | Meaning |
|---:|---|
| 0 | Visibility mask |
| 1 | Walls |
| 2 | Unresolved victims |
| 3 | Unresolved fires |
| 4 | Staging cells |
| 5-9 | Occupancy for each fixed agent slot |

`self_vec` layout:

| Indices | Meaning |
|---|---|
| 0-1 | x and y normalized by map extent |
| 2-4 | One-hot role: Scout, Ambulance, Fireman |
| 5 | Busy flag |
| 6 | Remaining service fraction |
| 7 | Previous move blocked/shortened flag |
| 8 | Previous interaction-started flag |
| 9-14 | One-hot previous executed physical action |
| 15 | Current tick normalized by horizon |

An incident being “discovered” is used for diagnostics only. There is no global
discovery map injected into actor observations and no discovery reward. A Scout
must learn to encode useful information into its eight-dimensional message.

## 7. Privileged critic state

The centralized critic sees:

| Key | Shape | Contents |
|---|---:|---|
| `global_grid` | `(9, H, W)` | Walls, staging, victims, fires, five agent slots |
| `agent_state` | `(5, 8)` | Presence, normalized position, role, busy state |
| `global_vec` | `(5,)` | Time, unresolved victim/fire fractions, team size, horizon |

The critic also receives a five-way one-hot identifying the agent whose value is
being estimated. The actor never receives this global state.

## 8. Communication channel

The channel is action-based, global-range, simultaneous, differentiable only
through the policy objective, and otherwise simulated as ordinary environment
state. Messages do not automatically contain coordinates or observations.

| Mode | Send gate | Destination | PPO includes |
|---|---|---|---|
| `nocomm` | Never sends | None | Physical action only |
| `always_broadcast` | Always | Every teammate | Physical + message |
| `when_broadcast` | Learned `send` | Every teammate when send=1 | Physical + send; message only when sent |
| `who_only` | Always | One learned teammate | Physical + recipient + message |
| `when_who` | Learned `send` | One learned teammate when send=1 | Physical + send; recipient/message only when sent |

Detailed behavior:

- The actor emits an eight-dimensional Gaussian mean and one shared, learned,
  state-independent log-standard-deviation vector. A sampled Gaussian is squashed
  with `tanh`; deterministic evaluation uses `tanh(mean)`.
- The environment clips external message values to `[-1, 1]`.
- A zero-extra-latency packet sent while resolving tick `t` is delivered in the
  observation returned for tick `t+1`. Configured extra latency adds whole ticks.
- Packet loss is sampled independently per sender-recipient attempt from the named
  channel RNG stream.
- Cost is charged per attempted recipient, including dropped packets. A broadcast
  by one agent costs `team_size - 1` attempts.
- Model-produced recipient actions are masked to present teammates other than self.
  An invalid recipient supplied externally produces an `invalid_recipient` event,
  no packet, and no attempt cost.
- The inbox holds at most four packets, enough for every other member of the
  five-agent team. Packets are sorted by fixed sender slot.
- Delivered messages are visible for that observation only; there is no persistent
  mailbox unless the recurrent policy remembers them.
- Physical dynamics use separate RNG streams, so channel loss/traffic cannot alter
  a paired physical trajectory when physical actions are fixed.

## 9. Reward and termination

One float32 team reward is computed and copied to every agent:

```text
reward = 10 * victims_resolved_this_tick
       + 10 * fires_resolved_this_tick
       + 20 if all incidents are now resolved
       - 0.02 per environment tick
       - communication_cost * recipient_attempts_this_tick
       - 2 * unresolved_incidents if the horizon is reached
```

The episode terminates successfully when every incident is resolved. It truncates
when the horizon is reached first. Discovery, movement, collision, distance, and
staging occupancy have no direct reward.

The V1 NoComm and AlwaysComm configs intentionally set communication cost to zero.
P1/P2/P3/O1 configurations use 0.01 per recipient attempt.

## 10. Procedural generation and determinism

The generator creates border walls, rectangular internal obstacles, a central 3 x
3 staging zone, clustered starting positions, and spatially separated incidents.
It rejects scenarios unless:

- every traversable cell is connected;
- incidents are sufficiently far from staging and invisible at reset;
- incidents have minimum Manhattan separation;
- victim/fire placement crosses quadrants;
- every incident is reachable by its specialist role;
- a deterministic privileged schedule fits within 70% of the horizon;
- wall count stays within the specified tolerance.

Five SHA-256-derived RNG streams isolate map, spawn, incident, tie-break, and
channel randomness. State hashes cover physical state, packets, channel RNG state,
and tie-breaking. Frozen pilot and P3 manifests each contain 8,000 train, 1,000
validation, and 1,000 test seeds with disjoint splits.

Training consumes manifest seeds in stable vector-index order and cycles after the
split is exhausted. Evaluation consumes an explicit contiguous slice and fails if
the requested slice exceeds the split.

## 11. Neural architecture

The implementation is parameter-sharing recurrent MAPPO with centralized training
and decentralized execution. Hidden size defaults to 128.

### Shared actor (376,893 parameters)

```text
local_grid 10x9x9
  -> flatten 810
  -> Linear 810->256 + ReLU
  -> Linear 256->128 + ReLU

self_vec 16
  -> Linear 16->32 + ReLU

each received packet [payload 8 + sender ID 5 + sender role 3] = 16
  -> Linear 16->64 + ReLU
  -> Linear 64->64 + ReLU
  -> masked mean over up to 4 packets

concat [128 grid + 32 self + 64 packet] = 224
  -> Linear 224->128 + ReLU
  -> GRUCell 128->128
  -> physical logits: 6
  -> send logits: 2
  -> recipient logits: 5
  -> message mean: 8, plus learned log std: 8
```

Physical and recipient logits are masked before categorical sampling. A separate
GRU hidden state is maintained for each environment-agent sequence and reset at
episode boundaries. All roles share the same weights; role and fixed agent slot
features let the network specialize behavior.

### Centralized critic (182,497 parameters)

```text
global_grid 9xHxW
  -> Conv2d 9->32, kernel 3, stride 2, padding 1 + ReLU
  -> Conv2d 32->32, kernel 3, stride 2, padding 1 + ReLU
  -> zero-pad encoded map to 32x6x6 and flatten = 1152

concat [grid 1152 + agent_state 40 + global_vec 5 + queried slot 5] = 1202
  -> Linear 1202->128 + ReLU
  -> Linear 128->128 + ReLU
  -> Linear 128->1
```

Total model size: **559,390 trainable parameters**.

The critic supports the included 11, 17, and 21-cell square presets. Although the
environment config accepts larger odd grids, the current critic rejects encoded
maps larger than 6 x 6 (roughly source dimensions above 24). Treat 21 x 21 as the
supported maximum unless the critic is revised.

Linear and convolution weights are orthogonally initialized; action heads use
gain 0.01, other layers use `sqrt(2)`, and biases start at zero.

## 12. MAPPO training

Default scientific hyperparameters:

| Setting | Value |
|---|---:|
| Environment timesteps | 2,000,000 |
| Rollout length | 128 |
| Learning rate | 0.0003 |
| Discount (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| PPO epochs | 4 |
| Minibatch target | 1,024 samples |
| PPO clip | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Gradient norm cap | 0.5 |
| Checkpoint interval | 100,000 environment timesteps |

`global_step` counts environment transitions, not per-agent decisions. One update
collects `rollout_steps * num_vector_envs` transitions. Training finishes after a
complete rollout, so the recorded final step can exceed `total_timesteps` by less
than one rollout batch. For example, the completed eight-environment V1 run ended
at 2,000,896 for a target of 2,000,000.

Rollouts are collected with the model on the requested device, retained on CPU,
then recurrent sequences are moved back to the device in PPO minibatches. The
environment workers and NumPy/PettingZoo simulation stay on CPU. Consequently,
CPU utilization is expected even with `device=cuda`, and this relatively small
model may not show sustained 100% GPU utilization.

The loss is clipped PPO policy loss plus clipped value loss minus entropy bonus.
Advantages use GAE and are normalized over the rollout. The communication-mode
table above defines which sampled heads contribute to log probability and entropy.

CUDA requests fail instead of silently falling back when CUDA is unavailable.
Strict deterministic PyTorch algorithms are enabled by default, with deterministic
cuBLAS workspace configuration on CUDA.

### Checkpoints and resume

Each atomic `.pt` checkpoint contains model, Adam optimizer, global step/update,
resolved config, config checksum, Python/NumPy/CPU/CUDA RNG states, complete vector
environment state, recurrent hidden state, episode returns, and episode lengths.

Resume requires an identical environment, identical training hyperparameters
except `total_timesteps` and `checkpoint_interval`, and the same total number of
vector environments. The worker packing may change if the total environment count
does not. Resume continues inside the original run directory.

An interruption does not force a new emergency checkpoint; resume uses the latest
completed interval checkpoint. Atomic temporary-file replacement prevents a
partially written checkpoint from masquerading as valid.

## 13. Evaluation

`relay-evaluate` loads model architecture and weights from a checkpoint, requires
the requested environment section to exactly match the trained environment, and
runs one scalar environment at a time. By default it uses deterministic actions
and the frozen test split.

It writes a full replay for every episode and reports per-episode return, length,
success, running aggregates, and ETA. It does not modify the training run. It
creates a new uniquely named evaluation run.

There is currently no validation scheduler, best-checkpoint selection, confidence
interval calculation, or aggregate comparison command. Those analyses must be run
separately from the saved JSON/Parquet data.

## 14. Runs, logging, and provenance

Run IDs use:

```text
YYYYMMDDTHHMMSSZ-experiment-name-first8_config_checksum
```

If the same ID already exists, `-01`, `-02`, and so on are appended. Existing runs
are not overwritten.

Each run contains:

```text
runs/<run-id>/
  manifest.json             status, command, timestamps, versions, provenance
  resolved_config.yaml      exact composed launch config
  config_diff.json          CLI overrides
  checkpoints/              interval checkpoints and latest.pt
  metrics/tensorboard/      TensorBoard scalar events
  metrics/evaluation.json   evaluation summary and episode records, when applicable
  events/                   Hive-partitioned zstd Parquet event stream
  replays/                  immutable per-episode evaluation replays
  errors/failure.json       exception record when a run fails
```

Training events include every timestep and PPO update. Evaluation events include
every timestep and episode. The timestep payload is intentionally detailed and can
consume hundreds of megabytes for a 2-million-step experiment; monitor free disk
space before running the full matrix.

Runs interrupted outside the managed exception path can remain marked `running`.
Treat a run as usable for resume only when it has a complete checkpoint. The
manifest status is provenance, not a live process monitor.

## 15. Replay semantics

A replay is a stored run, not a checkpoint being rerun by the viewer. Its Parquet
rows contain actions, rewards, done flags, infos, full snapshots, scenario seed,
checkpoint identity/step, config, and post-step hashes.

- `relay-replay view`: reads stored snapshots only; never advances the environment.
- `relay-replay verify`: creates a fresh seeded environment, reapplies stored joint
  actions, and compares every post-step SHA-256 state hash.
- `relay-replay generate`: evaluates a checkpoint once to create new stored replay
  artifacts.
- `relay-replay export`: renders PNG/GIF/MP4 from stored data.

The viewer supports seek, step, play/pause, speed, global or agent perspective,
overlay toggles, inspection data, file open, export, and replay generation.

## 16. Experiment matrix

| Config | Mode/change | Intended comparison |
|---|---|---|
| `v1_nocomm.yaml` | No communication, zero cost | V1 lower communication condition |
| `v1_always_comm.yaml` | Always broadcast, zero cost | V1 upper information condition |
| `p1_always.yaml` | Always broadcast, cost 0.01 | P1 reference |
| `p1_when.yaml` | Learned send gate + broadcast | P1 WHEN ablation |
| `p2_who_only.yaml` | Always send to learned recipient | P2 WHO ablation |
| `p2_full.yaml` | Learned send gate and recipient | P2 full WHEN+WHO |
| `p3_context.yaml` | Five-agent WHEN+WHO | Contextual recipient study |
| `o1_latency.yaml` | WHEN+WHO with 2 extra latency steps | Latency robustness |
| `o1_loss.yaml` | WHEN+WHO with 10% packet loss | Loss robustness |

Scientific configs lock environment identity, preset, grid, team, incidents,
sensors, motion, horizon, and communication mode against CLI overrides. Execution
settings, total timesteps, checkpoint interval, and other unlocked fields can be
overridden with typed `key=value` arguments. Unknown keys and invalid values fail
before a run directory is created.

The repository defines nine deterministic P3 evaluation fixtures: two Ambulance
busy cases, two Fireman busy cases, two informed-Ambulance cases, two relative-
distance cases, and one both-Ambulances-busy case. They are explicitly forbidden
for training.

## 17. Random and oracle baselines

The random policy samples a legal physical action, send bit, valid teammate, and
uniform payload every tick. Paired NoComm/AlwaysComm baseline runs use identical
seeds and identical physical-action RNG consumption, then assert equal physical
trajectory checksums. This isolates channel traffic from physical randomness.

The scripted oracle reads privileged state, assigns role-compatible specialists by
shortest path, and resolves incidents. It is a feasibility/calibration upper bound,
not a learnable policy and not evidence of decentralized performance.

## 18. Current local scientific artifacts

At audit time, this workspace contains:

- A completed CUDA V1 NoComm training run:
  `20260902T173250Z-v1-nocomm-f950b27b`, 2,000,896 steps, 1,954 updates.
- Its completed 100-episode test evaluation:
  `20260903T021039Z-v1-nocomm-280c951c`, success rate 23%, mean return
  36.1366, mean length 211.17.
- A completed 100-pair random NoComm/AlwaysComm baseline under
  `output/baselines`.
- Two older V1 run directories still marked `running`; their manifests indicate
  CPU launches that were interrupted. They have not been deleted by this audit.

No completed AlwaysComm learned-policy training run was found at audit time.

## 19. Audit findings and limitations

1. **No rescue transport:** victims and fires resolve in place. Staging has no
   completion mechanic.
2. **No collision dynamics:** agent overlap is legal and unpenalized.
3. **Global communication range:** only mode, latency, packet loss, and cost limit
   messages; geometry does not.
4. **No automatic semantic message:** the eight floats are learned from task reward.
5. **No periodic validation/best model:** evaluation is a separate command.
6. **Large event logs:** full training logging can dominate disk use.
7. **Rollout-boundary totals:** final steps can slightly exceed the target.
8. **Supported learned-map size:** included critic architecture supports included
   presets up to 21 x 21, despite looser environment-only dimension validation.
9. **Interrupted status:** a hard process termination can leave `status=running`.
10. **Determinism boundary:** exact checkpoint continuation captures all implemented
    RNG/state, but cross-version or cross-hardware bit identity should not be
    assumed without verification.
11. **Evaluation statistics:** success/return/length means are saved, but confidence
    intervals and statistical tests are not implemented.
12. **GPU profile:** the network is small and simulation is CPU-bound, so GPU
    utilization may be bursty even while all model tensors and optimization are on
    CUDA.

These are implementation characteristics, not hidden behaviors. Change them only
if the research design changes, because they affect comparability with frozen runs.

## 20. Verification evidence

The automated suite covers deterministic reset/replay, observation isolation,
line-of-sight, action masks, role restrictions, service timing, every communication
mode, channel RNG hashing, reward reconstruction, PettingZoo API compliance,
process/scalar parity, worker-layout seed invariance, checkpoint recovery, frozen
manifest integrity, scripted-oracle feasibility, replay round-trip verification,
partitioned event logging, and immutable GUI rendering/export.

Quick gate:

```powershell
& .\Activate-Relay.ps1
python ci\run_release_checks.py --skip-manifests
```

Full gate, including exhaustive validation of both 10,000-scenario manifests:

```powershell
python ci\run_release_checks.py
```

