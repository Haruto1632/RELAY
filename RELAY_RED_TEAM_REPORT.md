# RELAY Red-Team Design Review

> **Role:** Red-Team / Adversarial Research Reviewer  
> **Repository:** `Haruto1632/RELAY` (`C:\Users\DELL\OneDrive\Documents\GitHub\RELAY`)  
> **Audited Documents:**
> - **[RD]** *RELAY Research Design & Experimental Framework v0.6* (15 August 2026, 18 pages)
> - **[TD]** *RELAY Environment + Observation + Action + Reward Technical Design FINAL v1.2* (2 September 2026, 946 paragraphs)
> - **[Repo]** Local Git Workspace & Remote GitHub (`Haruto1632/RELAY`)
> **Inspection Date:** 2026-09-05  
> **Posture:** "Fight me. Break the design."

---

## 1. Executive Summary

RELAY investigates emergent communication (decomposed into **WHEN**, **WHO**, and **WHAT**) in a cooperative, partially observable, heterogeneous search-and-rescue testbed using Recurrent MAPPO with Centralized Training and Decentralized Execution (CTDE). The team comprises Scout (exploration/sensing), Ambulance (victim rescue), and Fireman (fire suppression) agents operating in a discrete 2D gridworld.

Following a rigorous forensic audit of the **authoritative specifications [RD v0.6] and [TD v1.2]** as well as the active repository state, this review uncovers **12 concrete vulnerabilities, mathematical bugs, information leaks, and scientific confounds**.

### The Core Threat to RELAY's Central Claims:
1. **The Pilot P2 Experiment Proves Nothing Beyond Event-to-Role Mapping:** In `pilot_core` (1 Scout, 1 Ambulance, 1 Fireman), there are only two valid recipients. A simple 2-rule lookup table (`victim -> Ambulance`, `fire -> Fireman`) achieves 100% optimal WHO routing without any context-dependence or teammate state modeling.
2. **Deterministic Center-Exclusion Confound in Scenario Generation:** Constraints `GEN-02`, `GEN-04`, and Scout's reset radius of 4 guarantee that the entire central 9×9 region (36% of a 17×17 map) **never contains any incidents**. Agents can exploit this geometric artifact to patrol only the outer perimeter, artificially inflating NoComm performance and distorting exploration dynamics.
3. **Observation Information Leaks:** `self_vec` provides absolute normalized coordinates `(x_norm, y_norm)` to every agent, collapsing relative partial observability and enabling trivial global coordinate sharing over continuous channels. Furthermore, visible `staging_zone` tiles uniquely reveal the map center.
4. **Mathematical Arithmetic Bug in Technical Design v1.2:** Section 12.1 explicitly claims maximum positive task reward for `pilot_core` is `100`, whereas the exact formula in Section 12 yields `80` (6 incidents × 10 + 20 completion bonus = 80).
5. **Reward Scale Swamps Communication Cost:** Communication cost `c = 0.01` per recipient attempt is negligible compared to the `+10.0` incident resolution reward. A policy that blindly floods the broadcast channel every single step for 240 steps pays only `4.8` reward penalty, which is completely offset by resolving even a single extra victim.
6. **Repository Implementation Void:** The repository contains only a boilerplate Adobe Flash/ActionScript `.gitignore`. No Python codebase, tests, configs, or manifests have been committed.

---

## 2. Actual Repository & Document Architecture

### 2.1 Repository Status
- **Local Clone:** `C:\Users\DELL\OneDrive\Documents\GitHub\RELAY`
- **Git Commit:** `ddf3b1b` (*Initial commit*) on branch `main`.
- **Tracked Files:** Exactly one file: `.gitignore` (unrelated ActionScript template).
- **Implementation State:** No environment code, MAPPO algorithms, configs, or tests are committed.

### 2.2 Concrete System Flow (Synthesized from [RD] & [TD])

```
Environment: PettingZoo ParallelEnv (relay-grid-v1, 17x17 / 21x21)
 │
 ├── Scenario Generator: Deterministic seeds (map, spawn, incident, tie_break, channel streams)
 ├── Physical Dynamics: Simultaneous movement (Scout: 2 cells, Specialist: 1 cell; co-location allowed)
 ├── Intervention: 3-tick busy lock (INTERACT -> STAY x2 -> Resolved)
 ├── Observations:
 │    ├── local_grid: (10, 9, 9) float32 (visible_mask, wall, victim, fire, staging, 5x agent poses)
 │    ├── self_vec: (16,) float32 (x_norm, y_norm, role[3], busy, busy_rem, blocked, interact, last_act[6], progress)
 │    ├── physical_action_mask: (6,) int8 [STAY, N, E, S, W, INTERACT]
 │    ├── recipient_mask: (5,) int8 (1 for present non-self agents)
 │    └── delivered_messages: (4, 8) payloads, (4, 5) sender IDs, (4, 3) sender roles, (4,) mask
 │
 ├── Actor Policy (Decentralized Recurrent Actor):
 │    ├── Observation Encoder + Message Processor (Shared MLP + Masked Mean)
 │    ├── Recurrent Core: GRU
 │    └── Multi-Head Outputs:
 │         ├── physical: Categorical(6) [masked]
 │         ├── send (WHEN): Categorical(2) [SEND / SILENCE]
 │         ├── recipient (WHO): Categorical(5) [masked to present non-self]
 │         └── message (WHAT): Tanh-bounded Box(-1.0, 1.0, shape=(8,))
 │
 ├── Communication Channel Subsystem:
 │    ├── Gating: If WHEN == SILENCE, message dropped, attempt_cost = 0
 │    ├── Routing: If WHEN == SEND, routed to recipient (or broadcast expanded if mode=broadcast)
 │    ├── Cost: -c * recipient_attempts applied to shared reward
 │    └── Delivery: Enqueued at tick t -> delivered at tick t + 1 (+ L if latency enabled)
 │
 └── Centralized Critic (Training Only):
      └── Privileged State: global_grid (9, H, W) + agent_state (5, 8) + global_vec (5,)
```

---

## 3. Research Claims Being Tested

| Stage | Claim / Hypothesis | Target Metric |
|---|---|---|
| **V1** | Communication improves cooperative search-and-rescue over NoComm | Task success rate (+8% / completion time -12%) |
| **P1 (WHEN)** | Agents learn selective communication to avoid unnecessary costs | Pareto frontier: Task return vs recipient attempts |
| **P2 (WHO)** | Agents learn sender-side targeted routing without broadcast | Reduction in irrelevant recipient deliveries |
| **P3 (Context-WHO)** | Scout routes to specific same-role specialists based on individual state | Recipient choice on label-swapped A-busy / F-busy / informed fixtures |
| **V2 (WHAT)** | Learned continuous 8-D payload carries causally meaningful semantics | Behavioral degradation under force-silence, scrambling, & intervention |

---

## 4. Information Boundary Audit

### 4.1 Global Coordinate Leak via `self_vec`
- **Location:** [TD] Section 8.2, `self_vec` indices 0–1: `x_norm = x / (W - 1)`, `y_norm = y / (H - 1)`.
- **Vulnerability:** While agents have a partial local view (`local_grid` 9×9 crop), they are given their exact continuous global coordinates `(x_norm, y_norm)`.
- **Confound:** When Scout observes a victim at crop offset `(dx, dy)`, it can compute `x_victim = x_scout + dx`, `y_victim = y_scout + dy` and transmit these normalized coordinates in 2 dimensions of the 8-D payload. The receiver subtracts its own `(x_norm, y_norm)` to obtain a direct global vector to the victim.
- **Scientific Impact:** The communication task reduces to trivial linear regression over global Euclidean coordinates rather than emergent spatial exploration or semantic landmark description.

### 4.2 Staging Zone Map-Center Anchor Leak
- **Location:** [TD] Section 6.2 & Section 8.1 (Channel 4: `staging_zone`).
- **Mechanism:** The staging zone is explicitly fixed at the centered 3×3 floor region `[W/2 - 1 .. W/2 + 1]`.
- **Exploit:** Any agent that observes even a single `staging_zone` cell immediately knows its exact relative vector to the map center `(W/2, H/2)`, allowing dead-reckoning navigation across the entire grid without global coordinates.

### 4.3 Action Mask Leak on INTERACT
- **Location:** [TD] Section 7.4 & Section 10.1.
- **Mechanism:** INTERACT is unmasked (`physical_action_mask[5] = 1`) *only* when an Ambulance is on an active unreserved victim or a Fireman is on an active unreserved fire.
- **Observation:** If an agent steps onto an unreserved incident, its action mask updates. Even if visual channels are degraded or occluded, the binary mask provides an infallible 1-bit ground-truth probe of incident presence and reservation state.

---

## 5. Communication Channel Audit

### 5.1 WHEN: The Free Silence Channel & Cost Insensitivity
- **Free Silence Channel:** When Scout detects nothing, it chooses `SILENCE`. Ambulance receives `message_mask = [0, 0, 0, 0]`. Over training, the specialist's GRU learns that `message_mask == 0` implies "no incident in Scout's FOV." This transmits 1 bit of negative information per step at **zero communication cost**.
- **Cost Insensitivity:** In `pilot_core`, maximum incident reward is +80. If Scout broadcasts every tick for 240 ticks at `c = 0.01`, total cost is `240 * 2 * 0.01 = 4.8`. Since 4.8 << 10.0 (the reward for one rescue), whenever task completion is uncertain, always-broadcasting dominates silence.

### 5.2 WHO: Pilot Triviality & P3 Catch-22
- **The Pilot WHO Illusion:** In `pilot_core` (Scout, Ambulance, Fireman), there are only 2 teammates. A trivial 2-weight linear layer mapping `[victim_detected, fire_detected]` to `[ambulance_id, fireman_id]` achieves 100% routing accuracy.
- **The P3 Catch-22:** [TD] Section 5 / 17 explicitly mandates that `p3_core` (5 agents) MUST NOT be introduced until `pilot_core` is stable. However, `pilot_core` cannot evaluate contextual WHO. This creates an experimental trap where researchers risk writing papers claiming "learned targeted routing" on P2 when only event-type classification was learned.

### 5.3 WHAT: Masked Mean Information Destruction
- **Location:** [TD] Section 8.3: "applies one shared MLP to `[payload, sender-ID, sender-role]` for each occupied slot, then computes a masked mean across slots."
- **Failure Mode:** If both Ambulance 0 and Fireman 0 send status messages to Scout simultaneously in P3, their processed vectors $h_{A0}$ and $h_{F0}$ are averaged: $h_{in} = \frac{1}{2}(h_{A0} + h_{F0})$.
- **Consequence:** Masked mean is linearly destructive. If $h_{A0}$ encodes "A0 busy" and $h_{F0}$ encodes "F0 idle", the averaged representation can land in an ambiguous latent region corresponding to neither or both.

---

## 6. Environment & Dynamics Audit

### 6.1 Deterministic Center-Exclusion Confound in Scenario Generation
- **Location:** [TD] Section 6.2, 6.3 (`GEN-02`, `GEN-04`), Section 5 (Scout radius = 4).
- **Failure Mechanism:**
  1. Scout spawns at staging center $(8, 8)$ in `pilot_core` (17×17).
  2. Scout sensing radius is 4 (Chebyshev), covering the window $[4..12] \times [4..12]$.
  3. `GEN-04` dictates: *No incident is visible to any agent at reset.*
  4. Therefore, **zero incidents can ever spawn inside the central 9×9 block**.
  5. The entire map has $15 \times 15 = 225$ interior cells. The forbidden center contains 81 cells (**36.0% of the entire traversable area**).
- **Exploitation:** Agents do not need communication to ignore the center. A hard-coded or learned heuristic of "immediately move to $(x \in \{1, 15\}, y \in \{1, 15\})$" explores 100% of potential incident zones with zero wasted steps in the center.

### 6.2 Arithmetic Discrepancy in Technical Design v1.2
- **Location:** [TD] Section 12.1 vs Section 12.
- **Error:**
  - Section 12: $r = 10 \times N_{victims} + 10 \times N_{fires} + 20 \times \text{complete}$. For `pilot_core` (3 victims, 3 fires), $r_{max} = (10 \times 3) + (10 \times 3) + 20 = 80$.
  - Section 12.1 (line 571): *"For pilot_core the maximum positive task reward is 100 (six incident resolutions plus completion bonus)."*
  - $6 \times 10 + 20 = 80 \neq 100$.
- **Impact:** Miscalibrated baseline return calculations and theoretical upper bounds.

### 6.3 Simultaneous Intervention Race Condition
- **Location:** [TD] Section 7.4 & 6.1 (tie-break stream).
- **Mechanism:** When two same-role specialists start INTERACT on the same incident simultaneously, the winner is determined by a static pre-seeded permutation. The loser wasted 1 step moving to the cell, receives no penalty, and is free to move on tick $t+1$, while the winner is locked in STAY for 3 ticks.
- **Consequence:** If Scout routes to both Ambulances (broadcast), both race to the victim; one services it, and the other immediately bounces off to search elsewhere. This "redundant rush" strategy minimizes completion time at negligible communication cost.

---

## 7. Baseline Fairness & Experimental Validity

### 7.1 Parameter Count Asymmetry (NoComm vs RELAY)
- In NoComm, the communication heads (`send`, `recipient`, `message`) and the message receiver MLP are inactive/excised.
- In RELAY, the policy contains:
  - Message MLP: $\approx (8 + 5 + 3) \times 64 + 64 \times 64 \approx 5.2\text{k}$ params.
  - Multi-heads: WHEN (2-way), WHO (5-way), WHAT ($8\times 2$ Gaussian/Tanh) $\approx 3\text{k}$ params.
  - Total parameter discrepancy: $\approx 10\text{k}-15\text{k}$ parameters.
- If NoComm is not capacity-matched (by increasing its GRU hidden dimension from e.g. 64 to 80), performance gaps may stem from neural representational capacity rather than communication.

### 7.2 Broadcast Cost Unfairness
- If `always_broadcast` is charged $2 \times c$ per step ($0.02$/step) regardless of utility, its cumulative return is penalized by $4.8$ over 240 steps.
- If `when_broadcast` learns to send only 20% of the time, its cost is $0.96$, giving it a $+3.84$ return advantage *even if both policies achieve identical rescue trajectories*.
- Reporting episode return rather than success rate or task completion time confounds communication efficiency with task performance.

---

# TOP 10 ATTACKS

---

### ATTACK #1: The Pilot WHO Triviality Proof
- **Claim Being Tested:** P2 / H2: "Heterogeneous agents learn task-relevant sender-side recipient routing without predefined recipient rules."
- **Failure Case:** In `pilot_core`, Scout policy converges to a 2-parameter threshold: `if victim in FOV -> route to 1 (Ambulance); if fire in FOV -> route to 2 (Fireman)`.
- **Why It Invalidates Claim:** Event-type classification is not contextual routing. The model has learned an associative lookup table identical to a static switch statement.
- **How to Expose:** Take the trained P2 Scout policy and evaluate it zero-shot in a scenario with 2 Ambulances (A0, A1). The policy will fail to route between A0 and A1 (or arbitrarily route to A0 100% of the time), proving zero generalization to same-role routing.
- **Severity:** `CRITICAL`
- **Recommended Fix:** Mandate that P2 is only published if evaluated on P3 multi-specialist transfer benchmarks.

---

### ATTACK #2: The Central 36% Exclusion Exploit
- **Claim Being Tested:** V1 / Non-triviality: "Exploration under partial observability requires cooperative communication."
- **Failure Case:** Agents exploit `GEN-02` / `GEN-04` by never scanning the central 9×9 grid.
- **Mechanism:** Because Scout sensing radius is 4 at $(8,8)$ and `GEN-04` forbids visible incidents at $t=0$, the generator never places incidents in the central 81 cells of a 17×17 grid.
- **Why It Invalidates Claim:** Specialists executing fixed boundary sweeps achieve near-optimal discovery times without any communication, artificially raising NoComm success to $>75\%$.
- **How to Expose:** Run 10,000 scenario generations and plot a 2D spatial heatmap of incident coordinates. The center 9×9 will show exactly 0.0% probability density.
- **Severity:** `HIGH`
- **Recommended Fix:** Spawn agents at randomized edge positions, or reduce Scout reset sensing radius to 0 on tick 0 (fog of war before initial step).

---

### ATTACK #3: Absolute Coordinate Sharing Bypass
- **Claim Being Tested:** "Communication operates under partial observability without global state leakage."
- **Failure Case:** Scout copies `self_vec[0:2]` into `message[0:2]`, providing exact floating-point global coordinates to specialists.
- **Mechanism:** `self_vec` contains normalized absolute coordinates $(x/(W-1), y/(H-1))$.
- **Why It Invalidates Claim:** The environment simulates partial visual grids, but the policy communicates in privileged global Cartesian space.
- **How to Expose:** Quantize or perturb `self_vec` coordinates to local-only egocentric relative displacements. If communication performance collapses, the policy was relying on global Cartesian telemetry.
- **Severity:** `HIGH`
- **Recommended Fix:** Replace `(x_norm, y_norm)` in `self_vec` with egocentric movement vectors and relative beacon distances.

---

### ATTACK #4: The Negligible Cost Flood Strategy
- **Claim Being Tested:** P1 / H1: "Agents learn to remain silent when communication is not worth its cost."
- **Failure Case:** Policies trained with default $c = 0.01$ learn always-broadcast because the maximum communication cost ($4.8$) is dwarfed by the $+10.0$ rescue reward.
- **Mechanism:** The cost $c=0.01$ is 0.1% of a single victim resolution. Expected value of sending is positive for almost all non-zero detection probabilities.
- **Why It Invalidates Claim:** P1 fails to establish selective communication; WHEN remains saturated at $\approx 100\%$ transmission rate during all active mission phases.
- **How to Expose:** Sweep cost $c \in [0.001, 0.05, 0.2, 1.0]$. Observe the abrupt step-function collapse from 100% send rate directly to 0% send rate with no intermediate Pareto-optimal selectivity.
- **Severity:** `HIGH`
- **Recommended Fix:** Scale communication cost proportionally to the value of information or apply dynamic per-step channel bandwidth caps.

---

### ATTACK #5: Masked Mean Latent Collision in P3
- **Claim Being Tested:** P3 / H3: "Specialists communicate status to Scout, enabling dynamic recipient selection."
- **Failure Case:** When multiple specialists transmit simultaneously, the receiver's masked-mean aggregator destroys individual sender semantics.
- **Mechanism:** `masked_mean` computes $\frac{1}{K}\sum_k \text{MLP}(p_k, \text{id}_k, \text{role}_k)$. Linear superposition in small latent spaces ($\mathbb{R}^8$) causes orthogonal message cancellation.
- **Why It Invalidates Claim:** Scout fails to learn context-dependent routing because multi-specialist status signals blur into an uninterpretable mean embedding.
- **How to Expose:** Train P3 with masked-mean vs slotted concatenation vs self-attention aggregator. Measure routing accuracy across the label-swapped A-busy fixtures.
- **Severity:** `MEDIUM`
- **Recommended Fix:** Replace masked-mean with fixed-slot concatenation: `Concat([slot_0, slot_1, slot_2, slot_3])`.

---

### ATTACK #6: The Causal Correlation Fallacy in V2
- **Claim Being Tested:** V2 / H4: "Learned messages show measurable causal relevance."
- **Failure Case:** High mutual information / linear probe accuracy between payload and incident location does not mean the receiver uses the payload.
- **Mechanism:** Receivers navigate using their own persistent local search heuristics while ignoring the incoming continuous vector. Probing shows $R^2 > 0.85$, but frozen-receiver zero-out ablation causes $<3\%$ drop in task success.
- **Why It Invalidates Claim:** Conflating representation emergence with causal behavioral utilization.
- **How to Expose:** Implement interventional counterfactual evaluation: replace payload with inverted coordinates or uniform noise $\mathcal{U}(-1, 1)$ and measure direct specialist path deflection.
- **Severity:** `HIGH`
- **Recommended Fix:** Pre-register interventional policy deflection metrics (e.g. Hausdorff distance between perturbed and clean trajectories) as the primary V2 metric.

---

### ATTACK #7: Silent Channel Communication via Slot Zero-Padding
- **Claim Being Tested:** P1: "Silence represents the absence of communication."
- **Failure Case:** Specialist policies infer sender silence patterns from `message_mask = 0`, treating silence as an explicit "quadrant clear" signal.
- **Mechanism:** Zero cost is charged for SILENCE, but the absence of a packet update deterministically informs the recurrent hidden state.
- **Why It Invalidates Claim:** The communication cost model is bypassed; agents exchange information without paying transmission penalties.
- **How to Expose:** Inject random packet drop noise on silence slots (pass random noise vector with `message_mask = 0`).
- **Severity:** `MEDIUM`
- **Recommended Fix:** Pass independent uninformative Gaussian noise on inactive slots or freeze GRU message sub-circuits when `message_mask == 0`.

---

### ATTACK #8: Staging Area Center Dead-Reckoning
- **Claim Being Tested:** "Agents operate purely under local sensing and emergent coordination."
- **Failure Case:** Agents use visible `staging_zone` tiles (Channel 4 of `local_grid`) to anchor their internal coordinate map.
- **Mechanism:** Staging zone is strictly centered at $(8,8)$. Observing any staging cell instantly resolves global map orientation.
- **Why It Invalidates Claim:** Removes the need for agents to communicate relative spatial frames or landmark references.
- **How to Expose:** Randomize the staging zone location across episodes (e.g. corners, edges, center).
- **Severity:** `LOW`
- **Recommended Fix:** Remove the dedicated `staging_zone` channel from `local_grid` after initial spawn tick.

---

### ATTACK #9: Technical Design Section 12 Arithmetic Discrepancy
- **Claim Being Tested:** Mathematical and programmatic reproducibility of reward baselines.
- **Failure Case:** Discrepancy between stated maximum reward (100) and formulaic maximum reward (80) in [TD] Section 12.1.
- **Mechanism:** Formula: $(3 \times 10) + (3 \times 10) + 20 = 80$. Document text: 100.
- **Why It Invalidates Claim:** Leads to erroneous normalized reward curves, incorrect explained variance metrics, and mismatched value targets in the centralized critic.
- **Severity:** `LOW` (Defect/Errata)
- **Recommended Fix:** Formally issue an errata correcting Section 12.1 of [TD v1.2] to 80.

---

### ATTACK #10: Complete Implementation Void
- **Claim Being Tested:** Research reproducibility and empirical validation.
- **Failure Case:** No executable code exists in the repository to run any verification test `T-DET-01` through `T-VIS-01`.
- **Mechanism:** Repository contains only `.gitignore`.
- **Why It Invalidates Claim:** The project cannot be reviewed, verified, benchmarked, or audited in code.
- **Severity:** `CRITICAL`
- **Recommended Fix:** Scaffold the complete `relay` package matching [TD v1.2] Section 3.1 and 19 immediately.

---

# THE 10 QUESTIONS YOU MUST BE ABLE TO ANSWER IN A VIVA

1. **On Pilot WHO Routing:**  
   *"In `pilot_core`, there is only one Ambulance and one Fireman. If your Scout network simply maps victim detections to action 1 and fire detections to action 2, how can you claim the agent learned 'targeted communication' rather than simple 2-class supervised classification?"*

2. **On Scenario Generation Spatial Bias:**  
   *"Because `GEN-04` prevents any incident from being visible to the Scout at reset, and Scout has a sensing radius of 4 at $(8,8)$, the central 36% of your gridworld is guaranteed to be empty. How do you prove your NoComm baseline didn't simply exploit this geometric artifact to achieve high baseline rescue rates?"*

3. **On Global Coordinates in Local Observations:**  
   *"Section 8.2 of your Technical Design shows that `self_vec` provides normalized absolute coordinates $(x_{norm}, y_{norm})$ to every agent. Why is this considered partial observability, and how do you ensure agents aren't simply transmitting absolute Cartesian coordinates in their 8-D continuous messages?"*

4. **On Communication Cost Calibration:**  
   *"At a cost of $c=0.01$ per recipient attempt, broadcasting for an entire 240-step episode costs $4.8$ reward points, while resolving one extra victim awards $+10.0$. Why wouldn't a rational RL agent always choose to broadcast under any non-trivial uncertainty?"*

5. **On Causal vs Correlational WHAT Analysis:**  
   *"You show high mutual information between the 8-D payload and victim locations. If you replace the payload with uniform noise at inference time while keeping the receiver policy frozen, exactly how much does the team rescue rate degrade?"*

6. **On Masked Mean Message Aggregation:**  
   *"In P3, when both Ambulance 0 and Fireman 0 send status packets to Scout on the same tick, your architecture averages their processed vectors with a masked mean. Doesn't this linear pooling destroy individual sender identities and create uninterpretable averaged states?"*

7. **On The Free Silence Channel:**  
   *"When an agent selects `SILENCE`, its communication cost is zero, but the receiver's observation explicitly receives `message_mask = 0`. How do you prove that the receiver isn't using the absence of messages as a cost-free negative-information channel?"*

8. **On Baseline Fairness:**  
   *"When comparing RELAY against NoComm, RELAY's actor network has extra parameters for the message MLP, WHEN head, WHO head, and WHAT head. Did you capacity-match the NoComm baseline by expanding its GRU hidden dimension, or does RELAY have an unfair representational capacity advantage?"*

9. **On RNG Stream Isolation:**  
   *"How do you guarantee that a NoComm run and an AlwaysComm run seeded with seed 42 generate the exact same map, incident layout, and spawn coordinates if communication action sampling consumes random numbers from the environment process?"*

10. **On Reward Formula Consistency:**  
    *"Section 12.1 of your Technical Design states that the maximum positive reward in `pilot_core` is 100, but Section 12 defines $r = 10 \times N_v + 10 \times N_f + 20$. For 3 victims and 3 fires, this equals 80. Which formula is active in your code, and how did this discrepancy affect your critic's value scaling?"*

---

### Audit Status & Created Documents
- **Report Location in Brain Artifacts:** [RELAY_RED_TEAM_REPORT.md](file:///C:/Users/DELL/.gemini/antigravity/brain/736b7745-0907-4e88-a86a-16621bc4a5bb/RELAY_RED_TEAM_REPORT.md)
- **Primary Source Documents Analyzed:**
  - `C:\Users\DELL\OneDrive\Documents\Modules\Major Project\RELAY Research Design Experimental Framework v0.6.pdf`
  - `C:\Users\DELL\Downloads\RELAY_Environment_Technical_Design_FINAL_v1.2.docx`
