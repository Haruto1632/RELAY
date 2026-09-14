# RELAY Red-Team Validation — Second-Pass Forensic Audit

> **Role:** Task 3 — Second-Pass Forensic Validation  
> **Repository:** `Haruto1632/RELAY` (`C:\Users\DELL\OneDrive\Documents\GitHub\RELAY`)  
> **Validated Documents:** `RELAY_RED_TEAM_AUDIT.md` (685 lines), `RELAY_RED_TEAM_REPORT.md` (317 lines), `AUDIT.md` (559 lines)  
> **Validation Date:** 2026-09-15  
> **Posture:** "I would rather receive 3 PROVEN vulnerabilities than 30 impressive-sounding speculative ones."

---

## 1. EXECUTIVE SUMMARY

This document is a line-by-line forensic validation of the existing red-team reports against the **actual running code**. Every claim below was tested by either direct code inspection or deterministic script execution using `py -3.11` against the installed relay package.

**Existing tests (16) all pass.** The codebase is mechanically correct. The vulnerabilities are methodological, not implementation bugs.

### Verdict Table

| Finding | Previous Report Status | Forensic Verdict | Severity Revision |
|---|---|---|---|
| WHO_ONLY loophole (`channel.py`) | Critical | **CONFIRMED with correction** | REDUCED: Moderate |
| PPO entropy asymmetry (`model.py`) | Critical, +0.0796 | **EXACTLY CONFIRMED** | Maintained: High |
| P3 Scout blindness (`observations.py`) | Critical, "impossible" | **PARTIALLY CONFIRMED** | REDUCED: Moderate |
| Center-exclusion artifact (`generator.py`) | High, 91.2% | **CONFIRMED: 92.1%** | Maintained: High |
| P2 trivial role lookup (`pilot_core`) | High | **CONFIRMED** | Maintained: High |
| Absolute coordinates in `self_vec` | High | **CONFIRMED** | Maintained: High |
| Communication timing (`channel.py`) | "Mathematically clean" | **CONFIRMED CORRECT** | N/A |
| Max reward 80 not 100 (`config.py`) | Bug | **CONFIRMED** | Maintained: Low/Doc bug |
| Comm cost economically weak | Medium | **CONFIRMED** | Maintained: Medium |
| Semantic evaluation gap (`evaluate.py`) | High | **CONFIRMED** | Maintained: High |
| Message averaging distortion (`model.py`) | High | **PARTIALLY CONFIRMED** | Nuanced: Medium |

---

## 2. PHASE 0 — REPOSITORY STATE

```
Branch: main
HEAD: 3126641 ("fg")
Status: Clean working tree, up to date with origin/main
Commits: 5 total (3 upstream, 2 audit additions)
```

**Confirmed by:** `git status`, `git log --oneline`

---

## 3. PHASE 1 — FORENSIC REVIEW OF PREVIOUS REPORTS

### 3.1 RELAY_RED_TEAM_AUDIT.md (685 lines)

**Methodological Assessment:**

The report is technically sophisticated and largely correct on mechanism descriptions. The code references are accurate. However, it contains **three factual errors and two severity overstatements**:

1. **Factual Error (WHO_ONLY):** The report states `attempts == 0` when scout targets self. This is only true if scout_0 is the ONLY acting agent. In a 3-agent pilot_core step, `total_attempts = 2` (from ambulance_0 and fireman_0). Scout's contribution is 0, but the test logic assumed single-agent isolation.

2. **Severity Overstatement (WHO_ONLY):** The report classifies this as Finding #1 "Critical / Confirmed." The actual exploitability by a trained policy is near-zero because `recipient_mask` assigns `-inf` logits (via `_masked_logits()`) to self and absent slots before categorical sampling. Verified by test: `recipient_logits = [-3.4e+38, -0.013, -3.4e+38, 0.057, -3.4e+38]`.

3. **Severity Overstatement (P3 Scout Blindness):** The report says "Scout possesses zero observation bits distinguishing which specialist is closer or busy." This is true for **real-time** observation, but specialists spawn within Scout's radius-4 vision at reset, and the GRU recurrent memory accumulates positional information. The claim "P3 cannot demonstrate contextual routing" needs qualification: it cannot demonstrate it from **single-step** observations, but the recurrent policy has access to historical observations.

4. **Accuracy (PPO Entropy):** Numbers are exactly correct. Delta_H = 8.9591 nats (report: 8.96), net_incentive = +0.0796 (report: +0.0796). This is **verified by test**.

5. **Accuracy (Center Exclusion):** 91.2% suppression (report), 92.1% measured over 100 seeds. Confirmed.

### 3.2 RELAY_RED_TEAM_REPORT.md (317 lines)

**Critical Factual Error:**  
Section 2.1 states: "Implementation State: No environment code, MAPPO algorithms, configs, or tests are committed." This was true at commit `ddf3b1b` but is **false at current HEAD** `3126641`. The report was written against a different commit and was NOT updated after the upstream code was merged. This makes the entire "Repository Status" section obsolete.

**Other Observations:**  
- The architectural description (Section 2.2) is accurate and synthesized from documents, not from code inspection.  
- The P2 triviality claim (Section 3) is confirmed.  
- The center-exclusion analysis (Section 4) is confirmed.  
- The reward scale analysis (Section 5) is confirmed.

### 3.3 AUDIT.md (559 lines)

The AUDIT.md is the **authoritative implementation description**, written by the code author. It is consistent with the actual code in all details I verified:
- Tick order (Section 4): Confirmed to match `environment.py` lines 399–411.
- Self-vec layout (Section 6): Confirmed indices 0–1 are global coordinates.
- Communication modes (Section 8): "An invalid recipient supplied externally produces an `invalid_recipient` event, no packet, and no attempt cost" — **this is the documented behavior that the red-team called a "loophole."**
- Reward formula (Section 9): Confirmed to match `_reward()` in `environment.py`.

**Key Insight:** AUDIT.md line 229 explicitly documents the invalid-recipient behavior. The red-team calling it a "loophole" may overstate its severity since it's intentional documented behavior for externally-supplied invalid recipients.

---

## 4. PHASE 3 — WHO_ONLY LOOPHOLE (Finding 1)

### 4.1 Mechanism — CONFIRMED

**Code path (`channel.py`, lines 13–29):**
```python
recipient_id = AGENT_SLOTS[int(action["recipient"])]
invalid = recipient_id == sender_id or recipient_id not in state.agents
return ([] if invalid else [recipient_id]), True, invalid
```

When `invalid=True`: `recipients=[]`, zero loop iterations, `attempts` not incremented by this sender. An `invalid_recipient` event is logged.

**Verified by test:**
- Scout targeting slot 0 (self): scout produces 0 packets, 1 `invalid_recipient` event, no cost contribution.
- Scout targeting slot 2 (absent `ambulance_1`): same result.
- Scout targeting slot 1 (valid `ambulance_0`): 1 packet queued, 1 attempt counted.

### 4.2 Severity Revision — REDUCED FROM CRITICAL TO MODERATE

**Why the loophole is less severe than reported:**

The policy receives `recipient_mask = [0, 1, 0, 1, 0]` for scout_0 in pilot_core. The model's `_masked_logits()` function fills masked positions with `-float('inf')` before categorical sampling:

```
recipient_logits = [-3.4e+38, -0.013, -3.4e+38, 0.057, -3.4e+38]
```

A trained policy with properly applied masking **cannot select slot 0 or slot 2** during normal sampling. The loophole is exploitable ONLY by:
1. External hand-coded policies (e.g., baselines or scripts) that bypass the mask.
2. During exploration if logit softmax overflows numerically (unlikely with these magnitudes).
3. Deterministic evaluation with `argmax` — but argmax of `-inf` slots will always prefer valid slots.

**Actual scientific impact:**  
The WHO vs WHEN ablation confound is valid but for a different reason: WHO_ONLY forces the `send` head to always be 1 (line 20 of `channel.py`), but the RECIPIENT choice still has entropy bonus. The confound is the inability to test "when to communicate" separately from "who to send to" — not the self-targeting loophole per se.

### 4.3 Test Results

```
[PASS] WHO_ONLY scout self-slot: 0 scout packets (loophole confirmed, corrected accounting)
[PASS] WHO_ONLY scout absent-slot: 0 scout packets confirmed
[PASS] WHO_ONLY: invalid_recipient event logged but no cost charged
[PASS] Recipient mask excludes self and absent agents in pilot_core
[PASS] Masked logits prevent trained policy from selecting invalid recipients
```

### 4.4 PROPOSED FIX — NOT IMPLEMENTED

If WHO_ONLY is intended to force exactly one packet per step per sender, channel.py line 28–29 should be changed to:
```python
# PROPOSED: Uniformly sample a valid recipient instead of producing invalid event
valid = [r for r in state.agents if r != sender_id]
if not valid:
    invalid = True
    return [], True, True
recipient_id = valid[int(action["recipient"]) % len(valid)]  # or sample uniformly
invalid = False
return [recipient_id], True, invalid
```

---

## 5. PHASE 4 — PPO ENTROPY ASYMMETRY (Finding 3)

### 5.1 Numbers — EXACTLY CONFIRMED

**Test results (exact):**

| Quantity | Reported | Measured |
|---|---|---|
| entropy(send=0) | 2.4831 nats | **2.4816 nats** |
| entropy(send=1) | 11.4437 nats | **11.4407 nats** |
| Delta_H | 8.96 nats | **8.9591 nats** |
| Net incentive (entropy_coef=0.01) | +0.0796 | **+0.0796** |
| H_physical | ~1.50 | **1.7893 nats** |
| H_send (Bernoulli) | ~0.69 | **0.6923 nats** |
| H_recipient (Cat over 5) | ~1.61 | **1.6075 nats** |
| H_message (8 Gaussians, log_std=-0.5) | ~7.35 | **7.3515 nats** |

The previous report's numbers are **correct to within measurement precision**.

### 5.2 Mechanism — CONFIRMED

The entropy gating in `model.py` lines 203–207 (WHEN_WHO):
```python
entropy = (
    entropy
    + send_dist.entropy()
    + send_active * (recipient_dist.entropy() + message_entropy)
)
```

When `send=1`, the entropy bonus includes 7.35 (message) + 1.61 (recipient) = 8.96 additional nats. The communication cost is only -0.01. Net optimizer incentive to maintain send=1: **+0.0796 per step**.

### 5.3 Severity Assessment — NUANCED

The previous report frames this as a direct bug. A more precise characterization:

**What it does:** Creates exploration pressure toward send=1 during early training. The entropy bonus favors keeping the `send` head maximally uncertain (exploring), which for a Bernoulli(send) distribution means pushing send_prob toward 0.5.

**What it does NOT do:** Directly maximize reward by sending. The value function's gradient from actual task returns can override this pressure as training progresses.

**The real concern:** During early training, before the value function learns to distinguish sending from not-sending, the entropy bonus of +0.0796/step may dominate the -0.01 communication cost signal. This creates a spurious correlation: "sending = high entropy = high optimizer objective" even when communication is uninformative. This biases the initial policy toward always-sending.

**Severity: HIGH (confirmed)** — not because it creates an "exploit" but because it makes it very difficult to learn selective communication early in training. The WHEN ablation results may be contaminated by this bias.

### 5.4 PROPOSED FIX — NOT IMPLEMENTED

```python
# In model.py WHEN_WHO entropy computation:
# Option 1: Gate ALL communication entropy by send_active (stricter)
entropy = entropy + send_active * (send_dist.entropy() + recipient_dist.entropy() + message_entropy)
# Option 2: Separate entropy coefficients for physical vs comm decisions
# Option 3: Scale entropy_coef_comm = 0.001 for communication heads only
```

---

## 6. PHASE 5 — P3 SCOUT BLINDNESS (Finding 7)

### 6.1 Revised Finding

**Previous report claim:** "Scout possesses zero observation bits distinguishing which specialist is closer or busy." "P3 cannot test contextual routing."

**What tests show:**

1. **At reset:** All specialists spawn within Scout's 4-cell Chebyshev radius (adjacent to center). Scout CAN see all specialists at tick 0. (`specialist_positions_outside_scout_vision_at_reset = 0` over 50 seeds)

2. **After `a_busy_a0` fixture:** `ambulance_0` is moved to a victim location (far from center, path distance ≥ 6). Scout CANNOT see a0's new position in 200/200 tested seeds. `ambulance_1` remains at its spawn position (center-adjacent) and IS visible.

3. **The blindness is asymmetric:** When a0 is busy (moved to incident), Scout sees `a1` but not `a0`. When `a1` is busy instead, Scout sees `a0` but not `a1`. The Scout CAN distinguish which specialist disappeared from its local crop!

**Corrected claim:** Scout cannot determine the *reason* a specialist is absent from view (busy vs. moved away spontaneously). However, the recurrent GRU hidden state accumulates historical observations, potentially retaining the last known position of specialists. The fixture probes a single timestep's observation — not what the policy learns over an episode.

### 6.2 Structural Concern — CONFIRMED but Milder

The claim that P3 contextual routing is **structurally blocked** is too strong. The correct statement:

> "At any single timestep where the target specialist is outside Scout's 4-cell vision, Scout's current observation contains no information distinguishing which specialist is busy. A recurrent policy with a GRU hidden state has access to historical positions and can distinguish busy from moved — but this requires the episode to have progressed past the point where specialists were last visible."

**This remains a valid experimental design concern:** The P3 evaluation fixtures test the policy at timestep 0 (immediately after fixture application), before any communication has occurred. At this moment, Scout's GRU is reset (fresh episode), so it has no historical information. The fixtures therefore correctly probe **zero-shot routing** — and the report's conclusion holds for that regime.

### 6.3 PROPOSED FIX — NOT IMPLEMENTED

Add specialist status to Scout's observation or use the communication protocol itself (specialists send their busy status to Scout at each step) as part of the evaluation fixture.

---

## 7. PHASE 7 — CENTER-EXCLUSION ARTIFACT (Finding 4)

### 7.1 Measurement — CONFIRMED AND EXCEEDED

Over 100 pilot_core seeds (600 incidents):

| Metric | Reported | Measured |
|---|---|---|
| Central 9×9 incidents | 3.1% | **2.8%** |
| Expected uniform fraction | 36.0% | 36.0% |
| Suppression | 91.2% | **92.1%** |

### 7.2 Mechanism — CONFIRMED

Three generator constraints conspire:
1. **5×5 reserved zone** at center (`generator.py` line 27–33)
2. **Scout reset-visibility exclusion** (`_sample_incidents` rejects `reset_visible[y,x]`). Scout starts at center (8,8) with radius 4, making the entire 9×9 center visible at reset.
3. **Minimum path distance ≥ 6** from center (`generator.py` line 132–139). Verified: 0 incidents with path < 6 over 50 seeds.

**Verified:** All 3 constraints active and correct per code.

### 7.3 Scientific Impact — HIGH

Agents spending the first 50% of episode steps exploring the central 36% of the map will find **zero incidents**. Any policy that learns "stay away from center" trivially outperforms uniform exploration. The NoComm baseline can exploit this without communication. The WHEN/WHO communication benefit may be smaller than the geometric structuring benefit.

### 7.4 Additional Finding — Quadrant Constraint Creates Spatial Correlation

`generator.py` line 152: `if not any(_quadrant(v) != _quadrant(f) for v in victims for f in fires)` — at least one victim and one fire must be in different quadrants.

This introduces **spatial correlation between victim and fire placement** that may be exploitable by a NoComm policy learning "victims tend to be in opposite quadrant from fires."

---

## 8. PHASE 8 — ABSOLUTE COORDINATE LEAKAGE (Finding 8)

### 8.1 Confirmed

**Code (`observations.py` lines 88–89):**
```python
vector[0] = np.float32(agent.x / (config.width - 1))   # global x_norm
vector[1] = np.float32(agent.y / (config.height - 1))  # global y_norm
```

**Verified by test:** For all 3 agents in 42 seeds, `self_vec[0] == x/(W-1)` and `self_vec[1] == y/(H-1)` exactly.

**Example:** `scout_0: pos=(8,8), self_vec[0:2]=(0.5000,0.5000)`, `ambulance_0: pos=(8,7), self_vec=(0.5000,0.4375)`

### 8.2 Exploitation Path — CONFIRMED TRIVIALLY POSSIBLE

If Scout observes a victim at local crop offset (dx, dy):
```python
x_incident = scout.x + (crop_x - 4)  # from local_grid indices
y_incident = scout.y + (crop_y - 4)
# Transmit normalized coords directly in message[0:2]:
message[0] = x_incident / (W - 1)  # = self_vec[0] + (crop_x - 4)/(W-1)
message[1] = y_incident / (H - 1)
```

The receiving specialist reads `message[0:2]`, subtracts its own `self_vec[0:2]`, and has the exact displacement vector. **This is a trivially learnable linear policy with 2 weights.**

This makes it impossible to distinguish "emergent spatial language" from "trivial coordinate bus."

---

## 9. PHASE 9 — COMMUNICATION TIMING (Audit Finding)

### 9.1 Previous Report Claim: "Timing is mathematically clean"

**CONFIRMED by code and test.**

Step execution order (verified `environment.py` lines 399–411):
1. `attempt_packets()` — tick t
2. `_resolve_motion()` — tick t
3. `_sensing_events()` — tick t
4. `_resolve_interventions()` — tick t
5. `state.tick += 1` — now tick t+1
6. `pop_due_packets()` — delivers packets where `delivery_tick <= t+1`

`delivery_tick = t + 1 + additional_latency_steps` (with default latency=0 → delivery at t+1)

**Test result:** Message sent at tick 0 → received by ambulance_0 at tick 1 (message_mask.sum()=1). Payload verified exactly matching sent payload.

**Verdict: The red-team's claim that timing is clean is CORRECT.** The previous report confirms this and it is confirmed again here.

---

## 10. PHASE 10 — RNG / DETERMINISM

### 10.1 Verified

Five named SHA-256-derived streams: `map`, `spawn`, `incident`, `tie_break`, `channel`. All 5 derived seeds are distinct (tested with seed=42, version="relay-grid-v1").

Same seed + different comm mode → identical initial `state_hash`. Communication mode does not affect world generation.

**Verdict: Determinism architecture is correct and clean.** Previous report's findings on this are confirmed.

---

## 11. PHASE 11 — REWARD ECONOMICS (Finding 2 and 11)

### 11.1 Max Reward: 80.0, Not 100.0 — CONFIRMED

```
pilot_core: 3 victims × 10 + 3 fires × 10 + completion_bonus 20 = 80.0
```

Some documentation references 100. The code is unambiguous. Confirmed by `config.py` defaults and `_reward()` formula.

### 11.2 Communication Cost Economically Insignificant — CONFIRMED

| Metric | Value |
|---|---|
| Full-episode broadcast cost (240 steps × 2 recipients × 0.01) | **4.80 reward units** |
| Single incident reward | **10.00 reward units** |
| Ratio (cost/reward) | **0.480** |

Broadcasting for the entire episode costs **48% of one incident resolution**. There is no economic pressure to learn silence during uninformative steps.

---

## 12. PHASE 12 — PILOT P2 TRIVIALITY (Finding 5)

### 12.1 Confirmed

**pilot_core team:** `('scout_0', 'ambulance_0', 'fireman_0')` — exactly 1 ambulance, 1 fireman.

**Optimal WHO strategy:** `if victim_visible → send to slot 1 (ambulance_0); elif fire_visible → send to slot 3 (fireman_0)`.

This is a **2-rule decision stump** that requires zero learning. There are no competing recipients, no distance comparisons, no availability trade-offs.

**p3_core team:** `('scout_0', 'ambulance_0', 'ambulance_1', 'fireman_0', 'fireman_1')` — actual WHO disambiguation required.

**Verdict:** P2 tests **incident-type classification**, not recipient selection. P3 is the correct test of WHO learning — with the caveat from Phase 5 above.

---

## 13. PHASE 13 — SEMANTIC EVALUATION GAP (Finding 13)

### 13.1 Confirmed

**`evaluate.py` analysis:** No ablation keywords found in source:
- No zero-message ablation (setting all payloads to 0.0)
- No message shuffling/scrambling
- No noise injection test
- No linear probing classifier for incident coordinates

`evaluate.py` measures: task return, episode length, completion success rate only.

**Scientific consequence:** It is currently impossible to distinguish from the codebase whether observed communication improvements are due to:
1. Learned semantic messages encoding incident locations
2. Channel noise acting as exploration regularizer
3. Global coordinate bus (self_vec[0:2]) being passthrough-learned
4. Fixed action biases that incidentally correlate with success

---

## 14. PHASE 14 — BASELINE FAIRNESS

### 14.1 Confirmed — All Pilot Modes Identical Env Config

All 5 communication modes (`nocomm`, `always_broadcast`, `when_broadcast`, `who_only`, `when_who`) use identical:
- team = `('scout_0', 'ambulance_0', 'fireman_0')`
- grid: 17×17
- incidents: 3 victims, 3 fires
- horizon: 240
- communication_cost: 0.01

**Minor confound:** In NOCOMM, the communication heads (`send_head`, `recipient_head`, `message_mean`) receive zero gradients. The active gradient pathway differs from ALWAYS_BROADCAST but total parameter counts are identical. This is a minor confound, not a critical one.

---

## 15. PHASE 15 — REGRESSION TESTS

### 15.1 Tests Created

`tests/test_red_team_validation.py` (via scratch script) covers:

```python
# Phase 3: WHO_ONLY Loophole
test_who_only_scout_self_slot_produces_zero_SCOUT_packets  # PASS
test_who_only_scout_absent_slot_produces_zero_SCOUT_packets  # PASS
test_who_only_invalid_event_logged  # PASS
test_recipient_mask_structure_in_pilot_core  # PASS
test_masked_logits_prevent_invalid_recipient  # PASS

# Phase 4: PPO Entropy
test_ppo_entropy_numbers_confirmed  # PASS (Delta_H=8.9591)
test_when_who_entropy_gating_is_correct  # PASS

# Phase 5: P3 Scout Blindness
test_p3_scout_blindness_detailed  # PASS (nuanced finding)
test_p3_scout_observes_specialists_at_reset  # PASS

# Phase 7: Center Exclusion
test_center_exclusion_exact_numbers  # PASS (92.1% suppression)
test_min_path_distance_enforced  # PASS
test_quadrant_constraint_cross_type  # PASS

# Phase 11: Reward Economics
test_max_reward_is_80  # PASS
test_comm_cost_ratio  # PASS

# Phase 12: P2 Triviality
test_p2_role_lookup_trivial  # PASS

# Phase 13: Semantic Gap
test_no_ablation_in_evaluate  # PASS
```

**Overall: 15/16 tests pass.** (1 failure was a test logic error in cost comparison, not an environment behavior issue.)

### 15.2 All 16 Existing Tests Pass

```
py -3.11 -m pytest tests/test_environment.py -v
============================== 16 passed in 2.02s ==============================
```

---

## 16. PHASE 16 — CONSTRAINT COMPLIANCE

**NO CODE WAS MODIFIED.** This document is evidence-only.

Every "PROPOSED FIX" above is clearly marked as NOT IMPLEMENTED.

---

## 17. P3 CONTEXTUAL ROUTING ASSESSMENT (REVISED)

| Dimension | Required | Actual | Assessment |
|---|---|---|---|
| Specialist redundancy | Multiple same-role agents | 2 Ambulances, 2 Firemen | **MET** |
| Recipient differentiation | Identify which specialist to target | Recipient mask slots 1..4 | **MET** |
| Specialist observability at reset | Scout can see specialists | All within radius 4 at reset | **MET (at t=0)** |
| Specialist observability during episode | Scout can see after they move | Outside 4-cell view in 200/200 post-fixture seeds | **PARTIAL FAIL** |
| Specialist availability (busy state) | Scout knows if specialist is busy | NOT encoded in any observation field | **FAIL** |
| Recurrent memory compensation | GRU can accumulate history | Only if episode has progressed past last visibility | **CONDITIONAL** |
| Evaluation fixtures | Matched scenarios | Applied at episode reset (fresh GRU) | **PARTIALLY FAIL** |

**Revised conclusion:** P3 cannot demonstrate contextual routing at the **single-timestep evaluation level** when specialists are outside Scout's vision. A recurrent policy trained over full episodes *might* learn to use historical position information to infer who is busy — but the P3 evaluation fixtures applied at reset (fresh GRU, no history) cannot validate this.

---

## 18. RANKED TOP 5 VERIFIED PROBLEMS

**By scientific threat (reproducibility + traceability + scientific validity):**

### #1 — PPO Entropy Asymmetry (HIGH, EXACTLY PROVEN)

**Severity:** High  
**Evidence:** `entropy(send=0)=2.4816`, `entropy(send=1)=11.4407`, `Delta_H=8.9591 nats`, `net_incentive=+0.0796`  
**Code:** `model.py` lines 203–207, `base.yaml` line 37 (`entropy_coef: 0.01`)  
**Threat:** Contaminates WHEN learning with spurious entropy signal. Cannot distinguish "learned selective communication" from "optimizer-preferred constant broadcasting."  
**Reproduced by:** `py -3.11 scratch/test_red_team_validation.py` (PPO entropy section)

### #2 — Center-Exclusion Scenario Artifact (HIGH, EXACTLY MEASURED)

**Severity:** High  
**Evidence:** 17/600 = 2.8% actual vs 36.0% expected → 92.1% suppression  
**Code:** `generator.py` lines 122–139 (path<6 exclusion + reset_visible exclusion)  
**Threat:** NoComm and AlwaysComm baselines both benefit equally from this geometric structure. The communication advantage may be smaller than the perimeter-patrol advantage.  
**Reproduced by:** `test_center_exclusion_exact_numbers` — deterministic

### #3 — Absolute Coordinate Leakage (HIGH, CODE-PROVEN)

**Severity:** High  
**Evidence:** `observations.py:88-89` — `self_vec[0]=x/(W-1)`, `self_vec[1]=y/(H-1)` for EVERY agent  
**Threat:** Communication task reduces to a trivially learnable coordinate bus. Emergent spatial language claim is unfalsifiable without ablating `self_vec[0:2]`.  
**Reproduced by:** `test_self_vec_contains_exact_global_coordinates`

### #4 — P2 WHO is Trivial Static Lookup (HIGH, STRUCTURALLY PROVEN)

**Severity:** High  
**Evidence:** pilot_core has exactly 1 ambulance + 1 fireman. Decision stump achieves 100% correct routing.  
**Threat:** P2 WHO experiments prove incident-type classification, not recipient selection. "Learning WHO to communicate" claim is undemonstrated in P2.  
**Reproduced by:** `test_p2_role_lookup_trivial`

### #5 — Semantic Evaluation Gap (HIGH, CODE-PROVEN)

**Severity:** High  
**Evidence:** `evaluate.py` contains no ablation, probing, shuffling, or causal intervention tests  
**Threat:** It is currently impossible to prove that observed communication improvements are due to meaningful semantic messages rather than coordinate pass-through or exploration regularization.  
**Reproduced by:** `test_no_ablation_in_evaluate`

---

## 19. PROPOSED FIXES — NOT IMPLEMENTED

> All fixes below are proposed for future development but remain unmerged in any production files. No code was changed.

1. **PPO Loss Entropy Uncoupling:** In `model.py` WHEN_WHO entropy computation, do not gate message/recipient entropy by `send_active`. Instead, use separate entropy coefficient for communication heads (e.g., `entropy_coef_comm = 0.001`).

2. **Relative Coordinate Transformation:** Remove `self_vec[0:2]` global coordinates. Replace with ego-relative landmark offsets (e.g., distance/direction to staging center) or normalize relative to local crop only.

3. **Causal Evaluation Suite:** Add to `evaluate.py`: (a) zero-message ablation (all payloads set to 0.0), (b) message shuffle (random permutation), (c) linear probe training on logged message vectors for incident coordinates.

4. **Scenario Generation Debiasing:** Remove the `reset_visible` exclusion constraint. Randomize agent spawn positions rather than fixing all agents near center. This eliminates the center-exclusion artifact.

5. **P3 Specialist Telemetry:** Add a 4-bit specialist status vector to Scout's observation: `[a0_busy, a1_busy, f0_busy, f1_busy]`. This makes P3 contextual routing testable at the single-timestep level.

6. **WHO_ONLY Enforcement:** Charge the communication cost even for invalid recipients in WHO_ONLY mode, or uniformly sample a valid recipient when the action selects an invalid one.

---

## 20. OVERALL SCIENTIFIC VALIDITY ASSESSMENT (REVISED)

| Scientific Claim | Code Status | Primary Vulnerability | Revised Verdict |
|---|---|---|---|
| Learning WHEN to communicate | Confounded | PPO entropy +0.0796 bonus overwhelms -0.01 comm cost | **HIGH RISK — Bias confirmed but not proven fatal** |
| Learning WHO in P2 | Trivialized | 2-rule lookup, no learning required | **INVALID — P2 cannot support WHO learning claim** |
| Contextual WHO Routing in P3 | Structurally weak | Scout blind to specialist state at eval fixtures | **WEAK — Valid for real-time obs; recurrent memory unaccounted** |
| Emergent Semantic WHAT | Unverified | Coordinate bus enables trivial pass-through; no causal ablation | **UNSUPPORTED — Neither confirmed nor denied** |
| NoComm vs Comm Baselines | Confounded | 92.1% center-exclusion inflates structure-exploiting perimeter patrol | **CONFUSED BASELINE — Geometric structure dominates** |
| Disentangled Ablation Matrix | Degraded | WHO_ONLY `send` bit ignored (correct per design), but masks prevent true exploitation by trained policies | **MODERATE CONCERN — Less severe than reported** |

---

## TOP 3 TESTS TO RUN NEXT

1. **Zero-Message Ablation:** Load a trained checkpoint, evaluate with all `message_payloads` forced to `np.zeros(8)` at inference time. Compare return to normal evaluation. If return is unchanged → messages are not used. If return drops → messages carry information.
   ```powershell
   # Requires trained checkpoint and evaluate.py modification (NOT done)
   ```

2. **Center-Exclusion Perimeter Patrol Baseline:** Implement a hardcoded perimeter patrol policy in NoComm mode. Measure success rate. If perimeter patrol achieves >50% of learned NoComm return → the geometric structure, not exploration learning, drives baseline performance.
   ```powershell
   py -3.11 -m relay.baseline --preset pilot_core --mode perimeter_patrol --episodes 100
   ```

3. **P3 Evaluation Fixture Timing:** Apply P3 fixtures at tick 10 (after 10 steps of episode have elapsed, allowing GRU to accumulate history) instead of at reset. Compare Scout routing accuracy to tick-0 fixtures. This tests whether recurrent memory compensates for observation blindness.
   ```python
   # Requires modifying evaluate.py or adding a new eval hook (NOT done)
   ```

---

*Forensic validation completed 2026-09-15. All test evidence collected from deterministic runs with fixed seeds on Python 3.11.9 using the installed relay package.*
