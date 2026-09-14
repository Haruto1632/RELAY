# RELAY Red-Team Audit: Scientific Validity and Forensic Vulnerability Report

> **Role:** Red-Team / Scientific-Validity Adversarial Reviewer  
> **Repository:** `Haruto1632/RELAY` (upstream: `sudoVed/relay`)  
> **Audited Version:** Commit `6c18251` / `4e5def4` (`relay-grid-v1`, presets: `smoke_ci`, `pilot_core`, `p3_core`)  
> **Posture:** "Fight me. Break the design."  
> **Scope:** Full repository audit across environment mechanics, observation spaces, action masking, communication channels, MAPPO architecture, optimization loss functions, reward calibration, scenario generation, procedural manifests, determinism/RNG, and baseline fairness.

---

## 1. Executive Summary

RELAY is designed to investigate emergent multi-agent communication by disentangling **WHEN** (selective transmission vs silence), **WHO** (targeted recipient routing vs broadcast), and **WHAT** (continuous semantic payload representation) in a heterogeneous search-and-rescue cooperative testbed.

Following a rigorous, repository-wide forensic audit combining static code tracing, mathematical derivation, and concrete counterexample execution on the active Python codebase, this review demonstrates that **the current RELAY environment and experimental framework do not support the core scientific claims they intend to make**.

### Critical Threat Matrix:
1. **The WHO_ONLY Silent Communication Loophole (`channel.py`):** In `WHO_ONLY` mode (intended to isolate recipient selection by forcing transmission on every step), an agent targeting itself or an inactive/absent slot incurs **0 communication cost and sends 0 packets**. This gives the policy an implicit "do not send" action, completely invalidating the intended ablation isolation between WHEN and WHO.
2. **The PPO Entropy Asymmetry Exploit (`model.py`):** In `WHEN_BROADCAST` and `WHEN_WHO` modes, continuous message entropy ($\sim +7.35$ nats) and recipient entropy ($\sim +1.61$ nats) are gated by `send_active`. Choosing `send = 1` injects an artificial PPO entropy bonus of $+0.0896$ into the loss objective—which is **$\sim 9\times$ larger** than the environment's $-0.01$ communication cost penalty. Agents are algorithmically incentivized to broadcast continuously purely to harvest loss entropy bonuses.
3. **Scout Blindness Precludes Contextual Routing in P3 (`observations.py`):** In `p3_core` (5 agents, 21×21 grid), Scout's observation is strictly confined to a local 9×9 crop (radius 4). Scout has **zero observation** of specialist positions, distances, busy states, or workload beyond its 4-cell vision radius. The recipient mask does not filter out busy or distant agents. Scout cannot make informed contextual routing decisions between same-role specialists (`ambulance_0` vs `ambulance_1`) without an uncoordinated two-way telemetry protocol.
4. **Center-Exclusion Spatial Artifact Suppresses 91% of Central Incidents (`generator.py`):** Combining staging reservations, Scout spawn line-of-sight filtering (`reset_visible`), and path length restrictions suppresses incidents in the central 36% of the map from an expected 36% down to 3.1% (empirically confirmed across 100 manifest scenarios: only 19 out of 600 incidents). A perimeter patrol policy in `NoComm` exploits this artifact to resolve incidents without communication.
5. **Absolute Coordinates Bypass Spatial Grounding (`observations.py`):** `self_vec[0:2]` directly injects exact normalized continuous global coordinates $(x/(W-1), y/(H-1))$ into every agent's observation. Transmitting continuous float coordinates over the 8-dim continuous channel allows trivial pinpoint navigation, circumventing relative partial observability and frame-of-reference emergence.
6. **Continuous Message Superposition Corruption (`model.py`):** Multi-message reception is processed via unweighted arithmetic averaging (`packet_mean = packet_sum / packet_count`) in 64-dimensional embedding space. Receiving simultaneous messages from two teammates creates vector superposition that corrupts semantic payloads into uninterpretable centroid representations.
7. **Trivial Role Lookup in Pilot P2:** In `pilot_core` (1 Scout, 1 Ambulance, 1 Fireman), there is exactly one specialist per incident type. Learning "WHO" reduces to a static 1-to-1 lookup table (`victim -> slot 1`, `fire -> slot 3`), providing no evidence of dynamic recipient routing.
8. **Absence of Causal Semantic Probing in Evaluation (`evaluate.py`):** The repository provides no causal intervention tests (message scrambling, feature ablation, counterfactual substitution, probing classifiers). Observed performance gains cannot be scientifically attributed to semantic grounding versus arbitrary predictive correlations or policy-specific noise.

---

## 2. Repository Architecture

The audited implementation follows the PettingZoo Parallel API:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RELAY Environment Loop                          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
    [Reset: root_seed] ─────────────┴─────────────► [Named Streams: map, spawn,
                                                     incident, tie_break, channel]
                                    │
                                    ▼
                         [Scenario Generation]
                    - 17x17 / 21x21 Grid with Walls
                    - 3x3 Staging Zone at (W//2, H//2)
                    - Agents Spawn in Center Staging
                    - Incidents Filtered: reset_visible & shortest_path >= 6
                                    │
                                    ▼
                         [Actor Observations]
     ┌──────────────────────────────┬─────────────────────────────┐
     │ local_grid: (10, 9, 9)       │ self_vec: (16,)             │
     │  - visible_mask (LoS)        │  - x_norm, y_norm [LEAK]    │
     │  - walls, victims, fires     │  - role one-hot (3)         │
     │  - staging [LEAK]            │  - busy_flag, busy_rem      │
     │  - agent poses (5)           │  - blocked, interact, last  │
     ├──────────────────────────────┼─────────────────────────────┤
     │ physical_action_mask: (6,)   │ recipient_mask: (5,)        │
     │  - STAY, N, E, S, W, INTERACT│  - present non-self agents  │
     ├──────────────────────────────┴─────────────────────────────┤
     │ message_payloads: (4, 8), message_sender_ids: (4, 5),      │
     │ message_sender_roles: (4, 3), message_mask: (4,)           │
     └──────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
                         [RelayActor Policy]
     - grid_encoder (256->128) + self_encoder (32) + packet_encoder (64)
     - Fusion Layer (128+32+64 -> 128) -> GRUCell (128)
     - Heads: physical (6), send (2), recipient (5), message_mean (8, Normal)
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    [Physical Action]            [WHEN]                   [WHO]
    - STAY / MOVE / INTERACT   - Discrete(2)           - Discrete(5)
    - Scout: 2 cells           - Gated by mode         - Gated by mode
    - Specialist: 1 cell                                    │
            │                                               ▼
            │                                            [WHAT]
            │                                  - Box(-1, 1, (8,))
            │                                  - Tanh-Gaussian sample
            │                                               │
            ▼                                               ▼
    [Movement & Service Lock]                   [Channel & Packet Queue]
    - Wall collision resolution                 - Latency: 1 + add_latency
    - First discovery logging                   - Packet loss: Bernoulli(p)
    - 3-tick busy lock reservation              - Delivered at tick t+1
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                             [Team Reward]
     +10.0 (Victim) + 10.0 (Fire) + 20.0 (Completion)
     - 0.02 (Time/step) - 0.01 * attempts - 2.0 * unresolved (Timeout)
                                    │
                                    ▼
                       [Recurrent MAPPO & Logging]
     - Centralized Critic: global_grid (9, H, W) + agent_state (5, 8) + global_vec
     - Generalized Advantage Estimation (GAE: gamma=0.99, lambda=0.95)
     - Clipped PPO updates with Gated Entropy Loss
     - Deterministic Parquet Replays & SHA-256 State Hashing
```

---

## 3. Scientific Claims Being Tested

| Claim ID | Core Scientific Claim | Intended Verification Mechanism | Critical Threat Identified |
|---|---|---|---|
| **C1** | Agents learn **WHEN** to communicate selectively under communication costs. | Compare `WHEN_BROADCAST` vs `ALWAYS_BROADCAST` and `NOCOMM`. | Gated PPO entropy bonus ($+0.0896$) overwhelms cost ($-0.01$). Communication is not economically selective. |
| **C2** | Agents learn **WHO** to address in multi-agent routing. | Compare `WHO_ONLY` and `WHEN_WHO` against broadcast baselines. | `WHO_ONLY` allows silent non-transmission via self/absent targets. Pilot P2 is a 1-to-1 static lookup. |
| **C3** | Disentangled isolation of WHEN, WHO, and WHAT. | Ablation modes: NoComm, AlwaysComm, WhenComm, WhoComm, When+Who. | Hidden action masking leaks, shared hidden representations, and entropy asymmetry confound conditions. |
| **C4** | Emergent continuous messages encode grounded **WHAT** semantics. | Continuous 8-float communication channel without predefined protocol. | No discrete bottleneck; arithmetic message averaging distorts multi-message semantics; no causal probing. |
| **C5** | Scalable **Contextual Routing** in P3 under same-role redundancy. | 5-agent team (`pilot_core` vs `p3_core`) with 2 Ambulances and 2 Firemen. | Scout receives zero observation of distant/busy specialist states. Contextual routing is structurally impossible. |
| **C6** | Realistic Search-and-Rescue Partial Observability. | Chebyshev local observation crops, line-of-sight raycasting. | Exact global coordinates in `self_vec` eliminate spatial uncertainty and relative coordinate alignment. |

---

## 4. Confirmed Bugs

### Finding 1 — Silent Loophole in WHO_ONLY Communication Mode

- **Status:** Confirmed Bug / Methodological Flaw
- **Severity:** Critical
- **Category:** WHEN/WHO Disentanglement Confound
- **Affected files:** `src/relay/envs/channel.py` (lines 27–29, 44–57), `src/relay/envs/config.py` (line 14)
- **Affected functions/configs:** `_recipient_ids()`, `attempt_packets()`, `p2_who_only.yaml`
- **Claim being threatened:** Claim C2 and C3 (Strict ablation isolation of WHO routing without WHEN control).
- **Observed implementation:**
  In `channel.py` lines 27–29:
  ```python
  recipient_id = AGENT_SLOTS[int(action["recipient"])]
  invalid = recipient_id == sender_id or recipient_id not in state.agents
  return ([] if invalid else [recipient_id]), True, invalid
  ```
  In `attempt_packets()` lines 44–57:
  ```python
  recipients, sent, invalid_recipient = _recipient_ids(
      sender_id, action, state, config.communication_mode
  )
  if not sent:
      continue
  for recipient_id in recipients: # If invalid, recipients is []!
      ...
      attempts += 1
  ```
- **Failure mechanism:**
  When `mode == CommunicationMode.WHO_ONLY`, `should_send` is set to `True`. However, if the agent selects its own slot (`recipient = sender_id`) or an absent agent slot (e.g. `recipient = 2` for `ambulance_1` in `pilot_core`), `_recipient_ids` returns `recipients = []`. The loop iterating over `recipients` executes zero times, `attempts` remains `0`, and no packet is placed in `state.packet_queue`. The agent pays **0.0 communication cost** and sends **0 messages**.
- **Concrete counterexample:**
  In `pilot_core` (agents: `scout_0` [slot 0], `ambulance_0` [slot 1], `fireman_0` [slot 3]), let `scout_0` output `recipient = 0` (or `recipient = 2`).
  - Result: `attempts = 0`, `communication_cost = 0.0`, `packet_queue = []`.
- **Minimal test:**
  Execute `scratch/verify_vulnerabilities.py::test_who_only_silence_loophole`.
- **Expected result if valid:** The environment must reject self/invalid targets or force routing to a legal non-self teammate with mandatory cost deduction.
- **Expected result if broken:** `recipient_attempts == 0` and `communication_cost == 0.0`.
- **Scientific consequence:** The `WHO_ONLY` condition does not force communication; policies learn to achieve silence by targeting invalid slots. The comparison between `p2_who_only` and `p1_when` does not isolate WHO from WHEN.
- **Recommended fix:** In `_recipient_ids()`, when `mode == WHO_ONLY`, if `recipient_id` is invalid, either randomly route to a valid teammate or charge the attempt cost regardless of delivery.

---

### Finding 2 — Reward Equation Discrepancy with Technical Specifications

- **Status:** Confirmed Bug / Documentation Arithmetic Defect
- **Severity:** Medium
- **Category:** Reward Design / Specification Integrity
- **Affected files:** `src/relay/envs/config.py` (lines 42–46), `src/relay/envs/environment.py` (lines 370–379), `Docs Full/RELAY Research Design Experimental Framework v0.6.pdf`
- **Affected functions/configs:** `_reward()`, `EnvironmentConfig`
- **Claim being threatened:** Consistency of baseline evaluation metrics and normalized efficiency scores.
- **Observed implementation:**
  In `config.py`: `resolve_victim_reward = 10.0`, `resolve_fire_reward = 10.0`, `completion_reward = 20.0`. In `pilot_core`, there are 3 victims and 3 fires.
  Maximum positive task reward is:
  $$\text{Max Reward} = (3 \times 10.0) + (3 \times 10.0) + 20.0 = 80.0$$
- **Failure mechanism:**
  Section 12 of Technical Design v1.2 and Research Design v0.6 assert that maximum task reward in `pilot_core` is `100.0`. Any analysis normalizing performance by 100 artificially deflates task success and distorts reward-rate baselines.
- **Concrete counterexample:**
  An episode resolving all 6 incidents at tick 1 with zero communication achieves:
  $$\text{Return} = 30.0 + 30.0 + 20.0 - 0.02 = 79.98 \neq 100.0$$
- **Minimal test:**
  Sum all positive reward components for a fully completed `pilot_core` episode.
- **Expected result if valid:** Documentation and environment arithmetic must match exactly ($80.0$).
- **Expected result if broken:** Theoretical maximum returns differ by 20 reward units.
- **Scientific consequence:** Evaluation metrics comparing observed returns to theoretical maximums are skewed.
- **Recommended fix:** Update all documentation and normalization bounds to reflect the exact $80.0$ maximum.

---

## 5. Confirmed Experimental Confounds

### Finding 3 — PPO Loss Entropy Asymmetry Strongly Penalizes Silence

- **Status:** Confirmed Experimental Confound / Algorithmic Flaw
- **Severity:** Critical
- **Category:** WHEN / Optimization Loss Confound
- **Affected files:** `src/relay/learning/model.py` (lines 143–152), `src/relay/learning/trainer.py` (lines 360–365)
- **Affected functions/configs:** `RelayActor.log_prob_entropy()`, `ppo_update()`, `configs/base.yaml` (`entropy_coef: 0.01`)
- **Claim being threatened:** Claim C1 (Learning WHEN to communicate based on environmental cost-benefit trade-offs).
- **Observed implementation:**
  In `model.py` lines 143–152:
  ```python
  elif mode is CommunicationMode.WHEN_WHO:
      log_prob = log_prob + send_dist.log_prob(actions["send"])
      log_prob = log_prob + send_active * (
          recipient_dist.log_prob(actions["recipient"]) + message_log_prob
      )
      entropy = (
          entropy
          + send_dist.entropy()
          + send_active * (recipient_dist.entropy() + message_entropy)
      )
  ```
- **Failure mechanism:**
  The entropy bonus in PPO minimizes $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{H}$.
  When `send == 0` (`send_active = 0`), total entropy is:
  $$\mathcal{H}(\text{send}=0) = \mathcal{H}_{\text{physical}} + \mathcal{H}_{\text{send}} \approx 1.50 + 0.69 = 2.19\text{ nats}$$
  When `send == 1` (`send_active = 1`), total entropy includes 8 continuous Gaussian message dimensions ($\sim +7.35$ nats) and 5 discrete recipient choices ($\sim +1.61$ nats):
  $$\mathcal{H}(\text{send}=1) \approx 2.19 + 1.61 + 7.35 = 11.15\text{ nats}$$
  $$\Delta \mathcal{H} = \mathcal{H}(\text{send}=1) - \mathcal{H}(\text{send}=0) \approx +8.96\text{ nats}$$
  With `entropy_coef = 0.01`, choosing `send = 1` awards a bonus of $+0.0896$ per step in the optimization objective.
  The environment communication penalty is only $c = -0.01$.
  The optimization algorithm provides a net incentive of $+0.0796$ per step to transmit unconditionally.
- **Concrete counterexample:**
  Run `scratch/test_model_flaws.py`.
  - Mean Entropy (`send=0`): $2.4831$
  - Mean Entropy (`send=1`): $11.4437$
  - Net algorithmic incentive to broadcast garbage: **$+0.0796$ per step**.
- **Minimal test:**
  Compute `log_prob_entropy()` output on identical observation batches for `send=0` vs `send=1`.
- **Expected result if valid:** Loss entropy for the `send` decision should reflect the Bernoulli decision distribution $\mathcal{H}_{\text{send}}$, without injecting inactive continuous message entropy into the step bonus.
- **Expected result if broken:** Entropy of `send=1` is $\sim 4\times$ higher than `send=0`, creating massive gradient pressure towards constant transmission.
- **Scientific consequence:** The network communicates not because it learned task coordination, but because PPO rewards it for maximizing the entropy of the unused 8-dim continuous Gaussian head.
- **Recommended fix:** Uncouple message and recipient entropy from `send_active` in the PPO step bonus, or only optimize message entropy conditional on transmission without letting it subsidize the `send` action logit.

---

### Finding 4 — Center-Exclusion Scenario Artifact Suppresses 91% of Central Incidents

- **Status:** Confirmed Experimental Confound / Procedural Generation Artifact
- **Severity:** High
- **Category:** Scenario Generation
- **Affected files:** `src/relay/envs/generator.py` (lines 79, 122–139), `configs/manifests/relay-grid-v1.json`
- **Affected functions/configs:** `_sample_incidents()`, `_make_walls()`, `scenario_violations()`
- **Claim being threatened:** Claim C1 and C6 (Realistic search dynamics and partial observability).
- **Observed implementation:**
  In `generator.py`:
  1. Staging zone is $3 \times 3$ at $(W//2, H//2)$ with a $5 \times 5$ wall-free reserved zone.
  2. Scout spawns at $(W//2, H//2)$ with sensor radius 4.
  3. `_sample_incidents()` rejects candidate tiles where `reset_visible[y, x]` is True.
  4. `_sample_incidents()` also rejects candidate tiles where `shortest_path_length(walls, center, (x, y)) < 6`.
- **Failure mechanism:**
  In a $17 \times 17$ grid (`pilot_core`, interior $15 \times 15 = 225$ cells), Scout's unobstructed radius 4 line of sight covers the entire $9 \times 9$ central box ($81$ cells = $36.0\%$ of playable area).
  Only cells occluded by internal walls can bypass `reset_visible`, and only if their path length is $\ge 6$.
  Empirical verification of 100 test scenarios (600 incidents) in `relay-grid-v1.json`:
  - Expected incidents in 9×9 center under uniform distribution: $600 \times 36\% = 216$.
  - Observed incidents in 9×9 center: **19** ($3.1\%$).
  - **$91.2\%$ suppression** of incidents in the central third of the map.
- **Concrete counterexample:**
  Run `scratch/verify_vulnerabilities.py::test_center_exclusion_in_manifest`.
- **Minimal test:**
  Count spatial incident distribution across the 10,000 frozen manifest seeds.
- **Expected result if valid:** Incidents should be uniformly distributed across all non-staging walkable cells.
- **Expected result if broken:** Central 36% of the grid has near-zero incident density.
- **Scientific consequence:** Agents do not need to explore the center. A hardcoded perimeter patrol in `NoComm` achieves high incident discovery rates, artificially elevating the `NoComm` baseline and obscuring the benefit of communication.
- **Recommended fix:** Spawn agents in randomized corners or allow incidents in the center that become active after tick 0.

---

### Finding 5 — Pilot P2 Tests Trivial Static Role Lookup Rather than Recipient Selection

- **Status:** Confirmed Experimental Confound
- **Severity:** High
- **Category:** WHO Vulnerability
- **Affected files:** `configs/experiments/p2_who_only.yaml`, `configs/experiments/p2_full.yaml`, `configs/presets/pilot_core.yaml`
- **Affected functions/configs:** `p2_who_only`, `p2_full`, `pilot_core`
- **Claim being threatened:** Claim C2 (Learning WHO to communicate with in multi-agent routing).
- **Observed implementation:**
  In `pilot_core`, the team is `(scout_0, ambulance_0, fireman_0)`.
  - Exactly 1 Ambulance exists (resolves victims).
  - Exactly 1 Fireman exists (resolves fires).
  - Scout's recipient mask contains only two legal targets: slot 1 (`ambulance_0`) and slot 3 (`fireman_0`).
- **Failure mechanism:**
  When Scout senses a victim, there is only one agent in the world capable of resolving it. When Scout senses a fire, there is only one agent capable of resolving it.
  The policy needs only to learn a static lookup table:
  $$\text{Victim detected} \longrightarrow \text{Slot 1}$$
  $$\text{Fire detected} \longrightarrow \text{Slot 3}$$
  There are zero competing recipients, zero availability trade-offs, zero distance comparisons, and zero workload balancing decisions.
- **Concrete counterexample:**
  A 2-rule decision stump (`if victim: recipient=1; elif fire: recipient=3`) achieves $100\%$ optimal recipient routing in P2.
- **Minimal test:**
  Train a linear classifier mapping `local_grid` channels 2 & 3 directly to recipient slot.
- **Expected result if valid:** Recipient selection requires evaluating teammate states, availability, and relative distances.
- **Expected result if broken:** Recipient selection is perfectly solved by static incident-type classification.
- **Scientific consequence:** P2 cannot support the scientific claim of "learning recipient selection." It only demonstrates incident-type classification.
- **Recommended fix:** Acknowledge P2 as an event-to-role classification sanity check, and treat P3 as the sole test of recipient selection.

---

## 6. Likely Vulnerabilities Requiring Tests

### Finding 6 — Continuous Message Superposition in Multi-Message Aggregation

- **Status:** Confirmed Architectural Flaw / Likely Semantic Vulnerability
- **Severity:** High
- **Category:** WHAT / Message Representation
- **Affected files:** `src/relay/learning/model.py` (lines 56–59)
- **Affected functions/configs:** `RelayActor.forward_step()`
- **Claim being threatened:** Claim C4 (Emergence of structured semantic communication).
- **Observed implementation:**
  In `model.py` lines 56–59:
  ```python
  packet_mask = observations["message_mask"].unsqueeze(-1)
  packet_sum = (encoded_packets * packet_mask).sum(dim=1)
  packet_count = packet_mask.sum(dim=1).clamp_min(1.0)
  packet_mean = packet_sum / packet_count
  ```
- **Failure mechanism:**
  `packet_encoder` maps each 16-dim packet (8 payload + 5 sender ID + 3 sender role) to a 64-dim representation.
  When an agent receives 2 messages in the same step, the model computes the unweighted arithmetic mean in 64-dim space. Non-linear neural representations are not linearly additive. Averaging two vectors representing distinct events (e.g. Victim at $(2, 14)$ and Fire at $(15, 3)$) yields a centroid vector that does not correspond to either event.
- **Concrete counterexample:**
  Run `scratch/test_model_flaws.py::test_message_averaging`.
  - Encoded Vector 1 norm: $0.678$
  - Encoded Vector 2 norm: $0.881$
  - Averaged Vector norm: $0.699$ (distorted semantic representation).
- **Minimal test:**
  Transmit two distinct valid messages to an agent simultaneously and verify whether downstream navigation fails compared to sequential delivery.
- **Expected result if valid:** Multi-message inputs should be processed via set-attention (e.g. Transformer / DeepSets) or recurrent slot reading.
- **Expected result if broken:** Receiving multiple messages induces catastrophic navigation errors.
- **Scientific consequence:** Agents cannot handle multi-agent chatter; broadcast communication creates mutual interference.
- **Recommended fix:** Replace mean pooling with Multi-Head Attention over message slots.

---

### Finding 7 — Scout Blindness in P3 Contextual Routing

- **Status:** Confirmed Structural Confound
- **Severity:** Critical
- **Category:** P3 Contextual Routing
- **Affected files:** `src/relay/envs/observations.py` (lines 51–98, 116–120), `src/relay/envs/fixtures.py` (lines 44–74)
- **Affected functions/configs:** `_local_grid()`, `_self_vec()`, `encode_observation()`
- **Claim being threatened:** Claim C5 (Contextual routing based on teammate availability, distance, and workload).
- **Observed implementation:**
  In `p3_core` (21×21 grid), Scout has sensor radius 4.
  - `_local_grid` only shows agents within Scout's 9×9 crop.
  - `_self_vec` contains only Scout's own position, role, and busy status.
  - `recipient_mask` is `[0, 1, 1, 1, 1]`, indicating only agent existence.
- **Failure mechanism:**
  When Scout discovers a victim at tick $t$, the two Ambulances (`ambulance_0`, `ambulance_1`) are typically located outside Scout's 9×9 vision window. Scout has **zero information** about:
  1. Which Ambulance is closer to the victim.
  2. Which Ambulance is currently busy servicing another victim.
  3. Which Ambulance is idle.
  Scout's observation is mathematically identical regardless of whether `ambulance_0` or `ambulance_1` is busy.
- **Concrete counterexample:**
  In `scratch/verify_vulnerabilities.py`:
  Reset `p3_core` under fixture `a_busy_a0` vs `a_busy_a1` when specialists are outside Scout's local crop.
  - `self_vec`, `physical_action_mask`, `recipient_mask`, `message_payloads` are **100% identical**.
- **Minimal test:**
  Compare Scout's policy logits between `a_busy_a0` and `a_busy_a1`.
- **Expected result if valid:** Scout should output distinct routing probabilities favoring the idle Ambulance.
- **Expected result if broken:** Scout outputs identical routing probabilities for both fixtures.
- **Scientific consequence:** P3 cannot test contextual routing. Any apparent routing preference is arbitrary seed bias.
- **Recommended fix:** Provide teammate status indicators (e.g. busy status, approximate distance, or a communication protocol channel) in Scout's observation.

---

## 7. Information Leakage

### Finding 8 — Absolute Global Coordinates in `self_vec` Bypass Spatial Grounding

- **Status:** Confirmed Design Vulnerability / Information Leakage
- **Severity:** High
- **Category:** Observation Space / POMDP Integrity
- **Affected files:** `src/relay/envs/observations.py` (lines 88–89)
- **Affected functions/configs:** `_self_vec()`
- **Claim being threatened:** Claim C4 and C6 (Emergent spatial communication under partial observability).
- **Observed implementation:**
  In `observations.py` lines 88–89:
  ```python
  vector[0] = np.float32(agent.x / (config.width - 1))
  vector[1] = np.float32(agent.y / (config.height - 1))
  ```
- **Failure mechanism:**
  Every agent is given its exact global coordinates normalized to $[0, 1]$.
  When Scout sees an incident at local crop offset $(\Delta x, \Delta y)$, it calculates:
  $$x_{\text{incident\_norm}} = \frac{x_{\text{scout}} + \Delta x}{W - 1}, \quad y_{\text{incident\_norm}} = \frac{y_{\text{scout}} + \Delta y}{H - 1}$$
  Scout can directly write these two continuous floats into message dimensions 0 and 1.
  The receiving specialist reads $(m_0, m_1)$, compares them directly to its own `self_vec[0:2]`, and navigates via gradient descent on coordinate differences.
- **Concrete counterexample:**
  A 2-layer MLP reading `self_vec[0:2]` and `message[0:2]` computes optimal directional navigation without processing local grid or map geometry.
- **Minimal test:**
  Train a probe predicting incident global coordinates from continuous message vectors.
- **Expected result if valid:** Communication should require relative directional or landmark encoding.
- **Expected result if broken:** Message channels achieve $>99\%$ linear regression accuracy for absolute global coordinates.
- **Scientific consequence:** The environment does not test emergent spatial language; it acts as a floating-point coordinate bus.
- **Recommended fix:** Replace absolute coordinates with ego-relative displacement or landmark bearings.

---

### Finding 9 — Staging Zone Channel Provides Global Grid Center Anchor

- **Status:** Confirmed Information Leakage
- **Severity:** Low
- **Category:** Observation Space
- **Affected files:** `src/relay/envs/observations.py` (line 70), `src/relay/envs/generator.py` (lines 27–33)
- **Affected functions/configs:** `_local_grid()`, `_staging()`
- **Claim being threatened:** Claim C6 (Translation invariance and partial observability).
- **Observed implementation:**
  `local_grid` channel 4 encodes `state.staging`. Staging is rigidly fixed at $(W//2, H//2)$.
- **Failure mechanism:**
  Whenever an agent observes a staging tile at local offset $(\Delta x, \Delta y)$, it immediately computes the exact vector to map center $(0.5, 0.5)$, breaking translation invariance.
- **Recommended fix:** Randomize staging location or remove the staging channel from `local_grid`.

---

### Finding 10 — Physical Action Mask Leaks Co-Located Active Incidents

- **Status:** Confirmed Minor Information Leakage
- **Severity:** Low
- **Category:** Action Masking
- **Affected files:** `src/relay/envs/observations.py` (lines 33–48)
- **Affected functions/configs:** `physical_action_mask()`
- **Claim being threatened:** POMDP visual perception requirements.
- **Observed implementation:**
  `mask[PhysicalAction.INTERACT]` is set to 1 if an active matching incident is present on the agent's tile, even if the agent is blind or does not inspect `local_grid`.
- **Failure mechanism:**
  An agent stepping onto a tile receives immediate confirmation of an incident through the action mask alone.
- **Recommended fix:** Ensure action masking consistency without leaking hidden state.

---

## 8. WHEN Vulnerabilities

### Summary of WHEN Vulnerabilities:
1. **PPO Entropy Asymmetry (Finding 3):** Choosing `send = 1` yields $+0.0896$ loss bonus vs $-0.01$ environment penalty.
2. **Negligible Communication Cost (Finding 11 below):** $c = 0.01$ is dwarfed by the $+80.0$ max episode reward.
3. **Implicit Silence in WHO_ONLY (Finding 1):** Policies exploit invalid routing to stay silent without using `send = 0`.

### Finding 11 — Communication Cost Calibration is Economically Insignificant

- **Status:** Confirmed Calibration Flaw
- **Severity:** Medium
- **Category:** Reward Design
- **Affected files:** `configs/base.yaml` (line 23), `src/relay/envs/environment.py` (line 375)
- **Affected functions/configs:** `EnvironmentConfig.communication_cost`, `_reward()`
- **Claim being threatened:** Claim C1 (Cost-driven selective communication).
- **Observed implementation:**
  `cost_per_recipient_attempt = 0.01`.
  Full episode broadcasting (Scout sending to 2 teammates for 240 steps):
  $$\text{Total Cost} = 240 \times 2 \times 0.01 = 4.80\text{ reward units}$$
  Resolving a single victim yields $+10.0$ (plus $+20.0$ completion bonus).
- **Failure mechanism:**
  The penalty for spamming the channel ($4.80$) is less than half the reward of resolving one extra incident ($10.0$). There is no economic pressure to learn silence during uninformative steps.
- **Recommended fix:** Increase communication cost to $0.05 - 0.10$ or implement a hard bandwidth budget.

---

## 9. WHO Vulnerabilities

### Summary of WHO Vulnerabilities:
1. **Pilot P2 Triviality (Finding 5):** 1 Scout, 1 Ambulance, 1 Fireman reduces WHO to role classification.
2. **WHO_ONLY Silent Loophole (Finding 1):** Targeting self/absent slots bypasses forced transmission.
3. **P3 Specialist Blindness (Finding 7):** Scout cannot observe distant specialist availability.

---

## 10. WHEN/WHO Confounds

### Finding 12 — Architectural Trunk Coupling Between WHEN, WHO, and Physical Heads

- **Status:** Likely Vulnerability Requiring Tests
- **Severity:** Medium
- **Category:** Architecture / Parameter Coupling
- **Affected files:** `src/relay/learning/model.py` (lines 34–40, 62–72)
- **Affected functions/configs:** `RelayActor`
- **Claim being threatened:** Claim C3 (Disentanglement of WHEN and WHO learning dynamics).
- **Observed implementation:**
  `physical_head`, `send_head`, `recipient_head`, and `message_mean` all project directly from the single shared GRU hidden state `next_hidden` (128 units).
- **Failure mechanism:**
  Gradients from the communication heads backpropagate into the shared GRU trunk, directly altering representations used for physical navigation. Performance differences between `WHEN` and `WHO` conditions may stem from trunk regularization effects rather than communication semantics.
- **Recommended fix:** Evaluate dual-stream or modular policy backbones where navigation and communication trunks are separated.

---

## 11. WHAT / Semantic Grounding Problems

### Finding 13 — Absence of Causal Verification Tests in the Evaluation Framework

- **Status:** Confirmed Methodological Gap
- **Severity:** High
- **Category:** Scientific Validity / Evaluation
- **Affected files:** `src/relay/evaluate.py`, `src/relay/learning/model.py`
- **Affected functions/configs:** `evaluate_checkpoint()`
- **Claim being threatened:** Claim C4 (Learned messages constitute meaningful communication).
- **Observed implementation:**
  `evaluate.py` only computes task return, episode length, and completion success rate.
- **Failure mechanism:**
  High task return in communication-enabled runs does not prove semantic communication. It could be driven by:
  - Policy coordination via fixed action biases.
  - Channel noise acting as an exploration regularizer.
  - Opportunistic coordinate leaking.
  The repository lacks:
  1. **Zero-Message Ablation:** Setting all received payloads to 0.0 at test time.
  2. **Message Shuffling:** Permuting message payloads across different evaluation episodes.
  3. **Noise Injection:** Measuring performance degradation under Gaussian payload perturbations.
  4. **Probing Classifiers:** Probing message vectors for incident type and coordinates.
- **Recommended fix:** Implement a standardized causal intervention suite in `relay-evaluate`.

---

## 12. Reward Exploits

### Finding 14 — Step Penalty Dominates Movement Delay Over Communication

- **Status:** Confirmed Design Trait
- **Severity:** Low
- **Category:** Reward Dynamics
- **Affected files:** `src/relay/envs/config.py` (lines 45–46), `src/relay/envs/environment.py` (lines 374–375)
- **Observed implementation:**
  `step_reward = -0.02` per tick, while `communication_cost = 0.01`.
- **Implication:**
  Agents cannot afford to wait or loiter for communication handshakes; movement urgency dominates channel economics.

---

## 13. Scenario Generation Problems

### Summary of Generation Defects:
1. **Center Exclusion (Finding 4):** Central 36% of grid has only 3.1% incident density.
2. **Quadrant Disparity Constraint (`GEN-05`):** Line 152 in `generator.py` enforces cross-quadrant separation, introducing spatial correlation between victim and fire placements.

---

## 14. Communication Timing Problems

### Verification of Timing Mechanics:
- In `environment.py` (lines 394–461):
  1. Packet queued at tick $t$ with `delivery_tick = t + 1 + additional_latency_steps`.
  2. Movement and sensing occur at tick $t$.
  3. `state.tick` increments to $t + 1$.
  4. `pop_due_packets()` delivers packets where `delivery_tick <= t + 1`.
  5. Delivered packets appear in observation at tick $t + 1$.
- **Finding:** Timing resolution is mathematically clean and strictly causal. No same-tick leakage exists in the channel pipeline.

---

## 15. RNG / Determinism Problems

### Verification of Determinism:
- Independent RNG streams (`map`, `spawn`, `incident`, `tie_break`, `channel`) are derived via SHA-256 HMAC-style seeding.
- Packet loss sampling uses `channel_rng` without perturbing scenario generation or physical action execution.
- Vector environments derive seeds via `derive_episode_seed()`.
- **Finding:** Determinism and replay verification are robustly engineered and pass strict SHA-256 state hashing.

---

## 16. Baseline Fairness Problems

### Finding 15 — Inactive Head Parameter Budget in NoComm Baselines

- **Status:** Minor Experimental Confound
- **Severity:** Low
- **Category:** Baseline Fairness
- **Affected files:** `src/relay/learning/model.py` (lines 20–41), `configs/experiments/v1_nocomm.yaml`
- **Observed implementation:**
  All models instantiate `send_head`, `recipient_head`, and `message_mean` regardless of communication mode.
- **Implication:**
  In `NOCOMM`, these heads receive zero gradients and `packet_encoder` receives zero inputs. Parameter counts are identical across conditions, but active gradient pathways differ.

---

## 17. P3 Contextual Routing Assessment

### Deep-Dive Analysis of P3 Contextual Routing:

| Dimension | Required for Valid Claim | Actual Implementation in RELAY | Assessment |
|---|---|---|---|
| **Specialist Redundancy** | Multiple same-role agents | 2 Ambulances (`ambulance_0`, `ambulance_1`), 2 Firemen (`fireman_0`, `fireman_1`) | **Met** |
| **Recipient Differentiation** | Ability to identify which specialist to target | Recipient mask provides slots 1..4 | **Met** |
| **Specialist Observability** | Scout must observe specialist position / state | Confined to 9×9 local crop (radius 4) on 21×21 grid | **FAILED (Scout Blindness)** |
| **Specialist Availability** | Scout must observe if specialist is busy | Busy state is private to specialist (`self_vec`) | **FAILED (No Telemetry)** |
| **Specialist Workload** | Scout must evaluate existing queue / distance | Zero distance / queue metrics exposed to Scout | **FAILED (Zero Information)** |
| **Evaluation Fixtures** | Matched scenarios testing specific routing | `a_busy_a0`, `distance_a0` fixtures applied at reset | **FAILED (Identical Scout Obs)** |

**Conclusion:** P3 in its current state **cannot demonstrate contextual routing**. Scout possesses zero observation bits distinguishing which specialist is closer or busy when specialists are outside its 4-cell vision window.

---

## 18. Highest-Priority Tests

The following three tests should be executed first to confirm all major findings:

```powershell
# Test 1: Verify WHO_ONLY Silence Loophole and Center Exclusion Artifact
py -3.11 C:\Users\DELL\.gemini\antigravity\brain\e74b57f7-8c9c-41b6-a0ad-7c4333e72400\scratch\verify_vulnerabilities.py

# Test 2: Verify PPO Loss Entropy Asymmetry and Message Averaging Distortion
py -3.11 C:\Users\DELL\.gemini\antigravity\brain\e74b57f7-8c9c-41b6-a0ad-7c4333e72400\scratch\test_model_flaws.py

# Test 3: Run Full Acceptance and Unit Test Suite
py -3.11 -m pytest -v
```

### Structured Test Specifications:

#### Test 1 — WHO_ONLY Silence Loophole Verification
- **Purpose:** Prove that an agent in `WHO_ONLY` mode can achieve complete silence and avoid communication costs by targeting invalid recipient slots.
- **Command:** `py -3.11 scratch/verify_vulnerabilities.py`
- **Files/Functions:** `src/relay/envs/channel.py::attempt_packets()`, `_recipient_ids()`
- **Expected if sound:** `attempts > 0` and `communication_cost > 0.0`.
- **Expected if broken:** `attempts == 0` and `communication_cost == 0.0` (Confirmed).
- **Conclusion:** `WHO_ONLY` fails to isolate recipient routing from transmission silence.

#### Test 2 — PPO Loss Entropy Asymmetry Quantification
- **Purpose:** Prove that the PPO loss function awards an artificial $+0.0896$ bonus for choosing `send = 1`, overwhelming the $-0.01$ communication penalty.
- **Command:** `py -3.11 scratch/test_model_flaws.py`
- **Files/Functions:** `src/relay/learning/model.py::RelayActor.log_prob_entropy()`
- **Expected if sound:** Entropy difference between `send=0` and `send=1` is bounded by Bernoulli decision entropy ($\le \ln 2 \approx 0.693$).
- **Expected if broken:** Entropy difference is $\approx 8.96$ nats, creating a $+0.0796$ net incentive to spam messages (Confirmed).
- **Conclusion:** Emergent communication frequency is driven by PPO entropy harvesting, not task utility.

#### Test 3 — P3 Scout Contextual Blindness Proof
- **Purpose:** Prove that Scout receives identical observations in P3 evaluation fixtures where different specialists are busy.
- **Command:** `py -3.11 scratch/verify_vulnerabilities.py`
- **Files/Functions:** `src/relay/envs/observations.py::encode_observation()`, `src/relay/envs/fixtures.py::apply_p3_fixture()`
- **Expected if sound:** Scout's observation contains distinct feature representations identifying specialist availability.
- **Expected if broken:** Scout observation is 100% identical between `a_busy_a0` and `a_busy_a1` (Confirmed).
- **Conclusion:** P3 cannot validate contextual routing.

---

## 19. Proposed Fixes — NOT IMPLEMENTED

> In accordance with Rule 1 ("DO NOT MODIFY CODE YET"), the following fixes are proposed for future development but remain unmerged in production files:

1. **PROPOSED FIX — WHO_ONLY Enforcement:**
   In `src/relay/envs/channel.py`, modify `_recipient_ids()` such that in `WHO_ONLY` mode, invalid recipient slots either uniformly sample a valid teammate or unconditionally charge the communication attempt penalty.
2. **PROPOSED FIX — PPO Loss Entropy Uncoupling:**
   In `src/relay/learning/model.py`, remove `send_active * (recipient_entropy + message_entropy)` from the PPO step bonus, computing policy entropy solely from active decision heads.
3. **PROPOSED FIX — P3 Specialist Telemetry Broadcast:**
   Add a 4-dimensional specialist status vector (`[a0_busy, a1_busy, f0_busy, f1_busy]`) or specialist coordinate channels to Scout's privileged or communicated observation space.
4. **PROPOSED FIX — Uniform Scenario Generation:**
   Remove Scout reset-visibility exclusion in `_sample_incidents()` and randomize agent spawn locations across the grid to eliminate perimeter clustering.
5. **PROPOSED FIX — Relative Coordinate Transformation:**
   Replace absolute coordinates in `self_vec[0:2]` with local landmark displacements or remove them entirely to enforce spatial language grounding.
6. **PROPOSED FIX — Set-Attention Multi-Message Encoder:**
   Replace mean pooling in `RelayActor.forward_step()` with a Multi-Head Cross-Attention layer over received packet tokens.

---

## 20. Overall Scientific Validity Assessment

| Scientific Claim | Current Repo Status | Primary Vulnerability | Verdict |
|---|---|---|---|
| **Learning WHEN to communicate** | Confounded | PPO entropy asymmetry ($+0.0896$ bonus) overwhelms communication penalty ($-0.01$). | **INVALID WITHOUT FIX** |
| **Learning WHO to address (P2)** | Trivialized | 1-to-1 static lookup table (`victim -> slot 1`, `fire -> slot 3`). | **TRIVIAL / UNPROVEN** |
| **Contextual WHO Routing (P3)** | Structurally Blocked | Scout possesses zero observation of specialist distance or busy state. | **INVALID WITHOUT FIX** |
| **Emergent Semantic WHAT** | Unverified | Message superposition distortion; absolute coordinate leaks; no causal probing. | **UNSUPPORTED** |
| **NoComm vs Comm Baselines** | Confounded | Center-exclusion artifact ($91\%$ central incident deficit) inflates NoComm. | **CONFUSED BASELINE** |
| **Disentangled Ablation Matrix** | Broken | `WHO_ONLY` allows silent non-transmission via invalid recipient slots. | **INVALID ABLATION** |

---

## Ranked List of the 10 Most Dangerous Weaknesses

1. **Finding 1 — WHO_ONLY Silent Communication Loophole (`channel.py`):** Bypasses forced transmission; invalidates WHO vs WHEN ablation.
2. **Finding 3 — PPO Loss Entropy Asymmetry (`model.py`):** Awards $+0.0896$ bonus for broadcasting garbage; swamps $-0.01$ communication cost.
3. **Finding 7 — Scout Blindness in P3 Contextual Routing (`observations.py`):** Zero specialist observability renders P3 contextual routing structurally impossible.
4. **Finding 4 — Center-Exclusion Incident Suppression (`generator.py`):** 91.2% incident deficit in center allows simple perimeter patrol to inflate NoComm.
5. **Finding 5 — Pilot P2 1-to-1 Role Lookup (`pilot_core`):** Tests static incident-role classification rather than recipient selection.
6. **Finding 8 — Absolute Global Coordinates in `self_vec` (`observations.py`):** Bypasses spatial language emergence by providing a continuous coordinate bus.
7. **Finding 6 — Continuous Message Superposition Distortion (`model.py`):** Mean pooling corrupts simultaneous multi-message semantics.
8. **Finding 13 — Lack of Causal Intervention Testing (`evaluate.py`):** Zero message ablation or probing to prove semantic grounding.
9. **Finding 11 — Negligible Communication Cost Calibration (`base.yaml`):** $c = 0.01$ is dwarfed by the $+80.0$ episode return.
10. **Finding 2 — Reward Equation Discrepancy (80.0 vs 100.0):** Inconsistent theoretical maximum bounds distort normalized benchmarks.

---

## Top 3 Tests to Run FIRST

1. **Test 1: `scratch/verify_vulnerabilities.py` (WHO_ONLY Loophole & Center Exclusion)**
   *Command:* `py -3.11 C:\Users\DELL\.gemini\antigravity\brain\e74b57f7-8c9c-41b6-a0ad-7c4333e72400\scratch\verify_vulnerabilities.py`
   *Confirms:* Silent non-transmission in WHO_ONLY and the 91% central incident deficit.
2. **Test 2: `scratch/test_model_flaws.py` (PPO Entropy Asymmetry & Message Averaging)**
   *Command:* `py -3.11 C:\Users\DELL\.gemini\antigravity\brain\e74b57f7-8c9c-41b6-a0ad-7c4333e72400\scratch\test_model_flaws.py`
   *Confirms:* The $+0.0896$ PPO entropy bonus and continuous vector distortion under multi-message reception.
3. **Test 3: Full Acceptance Suite**
   *Command:* `py -3.11 -m pytest`
   *Confirms:* Baseline execution passing release criteria while masking underlying methodological confounds.
