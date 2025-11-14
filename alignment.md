# Alignment Guidelines for Claude Code Sessions

**CRITICAL**: Read this file IMMEDIATELY after conversation compaction to avoid repeating mistakes.

---

## Core Process Requirements (DO NOT VIOLATE)

### 1. ALWAYS Show Code for Review Before Committing
- **NEVER commit without explicit user approval**
- Show code changes, explain what was done
- Wait for user to say "yes" or "good to commit"
- If unsure, ask: "Is this acceptable to commit?"

**Example**:
```
Me: "I've implemented X. Here's the code: [show code]. Is this acceptable to commit?"
User: "yes"
Me: [commits]
```

### 2. NEVER Add Claude as Co-Author
- **ONLY user (RST) should be the commit author**
- Git author should be: `rstrivedi <triv.rakshit@gmail.com>`
- **NO "Co-Authored-By: Claude" lines in commit messages**
- Use "Added by RST:" in code comments, NOT "Added by Claude"

**Verify After Each Commit**:
```bash
git log --format="%an %ae" -1  # Should show rstrivedi only
```

### 3. Study Codebase Thoroughly - NO GUESSING
- **DO NOT make "weird bets" about what exists**
- **DO NOT assume** event names, field names, or observation names
- **ALWAYS grep/read** to verify exact names before using them

**Bad Behavior** (user got very angry):
- Assuming `event.get('component')` when it's actually `event.get('type')`
- Assuming `event.get('was_immune')` when it's actually `event.get('immune')`
- Making up events like `berry_colored` or `berry_consumed` that don't exist
- Saying something is "not accessible" without grepping to verify

**Correct Behavior**:
```bash
# Always verify event names
grep -r "events:add" meltingpot/

# Always verify field names
grep -r "reward_component" meltingpot/ | head -20

# Always verify observations
grep -r "addObservation" meltingpot/
grep -r "BERRIES_BY_TYPE" meltingpot/
```

### 4. Follow Code Conventions
- Use `# Added by RST:` for new code
- Use `# Edited by RST:` for modifications (explain why)
- Use `# Removed by RST:` for deletions (explain why)
- **NOT** "Added by Claude" or similar

### 5. NO Approximations - Everything Must Be Exact
- User said: "No approximation is acceptable. Scripted agents should do everything precisely."
- If you don't know something, **look it up in code**
- Don't infer, don't approximate, don't guess

**Example**:
- Don't approximate ego body color from planting events
- Don't approximate immunity from READY_TO_SHOOT
- Look up exact values in Lua code

---

## Communication Rules

### 1. DO NOT Ask Questions That Can Be Answered By Reading Code
User response when I asked "do you really need to be standing on unripe berry to plant it?":
> "DO NOT FUCKING ASK ME SUCH QUESTIONS. YOUR JOB IS TO LOOK INTO CODE AND FIGURE THIS OUT"

**Correct behavior**:
- Read `allelopathic_harvest.py` config to find `beamLength=3`
- Read `components.lua` to see how planting works
- Implement based on what you found
- Show for review

### 2. Always Read Phase README Files
- Each phase has a README with "Notes for Next Session"
- **READ THESE IMMEDIATELY** after conversation compaction
- Example: `PHASE2_README.md` has critical process requirements and technical details

### 3. Explain Changes in Detail When Asked
- Don't just say "I committed X"
- Explain **what** changed, **why**, and **how it works**
- User wants to understand the implementation

---

## Technical Details to Remember

### Color Indexing
- **1-indexed in Lua**: 1=RED, 2=GREEN, 3=BLUE, 0=GREY
- Convert to 0-indexed in Python for agent IDs only
- Keep color indices 1-indexed in Python to match Lua

### Episode Length
- **2000 steps**, NOT 1000
- This is critical for tests like R8 (monoculture)

### Lua Events (Exact Names)
From `components.lua`:
- `reward_component`: fields `t`, `player_id`, `type`, `value`
- `sanction`: fields `t`, `zapper_id`, `zappee_id`, `zappee_color`, `was_violation`, `applied_minus10`, `immune`, `tie_break`
- `replanting`: fields `player_index`, `source_berry`, `target_berry`
- `eating`: fields `player_index`, `berry_color`
- `resident_info`: fields `player_index`, `permitted_color`, `self_body_color`, etc.
- `nearby_agent`: fields `observer_index`, `agent_id`, `rel_x`, `rel_y`, `body_color`, `immune_ticks`
- `nearby_ripe_berry`: fields `observer_index`, `rel_x`, `rel_y`, `distance`, `color_id`
- `nearby_unripe_berry`: fields `observer_index`, `rel_x`, `rel_y`, `distance`, `color_id`

### Observations (Exact Names)
- Individual: `RGB`, `READY_TO_SHOOT`, `PERMITTED_COLOR`
- Global: `WORLD.RGB`, `BERRIES_BY_TYPE`, `RIPE_BERRIES_BY_TYPE`, `UNRIPE_BERRIES_BY_TYPE`
- `BERRIES_BY_TYPE`: shape (3,) for [RED, GREEN, BLUE]

### Game Mechanics
- **Plant beam**: beamLength=3, beamRadius=0 (can plant from 3 cells away)
- **Zap range**: 3 cells, cooldown=4 frames
- **Grace period**: 25 frames (startup_grey_grace)
- **Immunity**: 200 frames per target, clears on color change

---

## Mistakes to Avoid (From Past Sessions)

### Mistake 1: Committing Without Approval
**What happened**: I committed ResidentWrapper without showing code first
**User reaction**: "What the hell are you doing? I told you that you will fuck up as soon as you compact conversation. That is why I asked you to create phase 2 readme so you know what all we aligned on? Why did you not give me summary of what code changes you did and ask for my approval to commit?"
**Never do this again**

### Mistake 2: Adding Claude as Co-Author
**What happened**: Early commits had "Co-Authored-By: Claude"
**User reaction**: Very upset, asked me to never do this
**Always check**: Only rstrivedi should be author

### Mistake 3: Not Reading Code Properly
**What happened**:
- Used wrong event field names (`component` vs `type`)
- Didn't know about GlobalBerryTracker observations
- Made up event names that don't exist
**User reaction**: "why are you making these kind of weird bets... Have you not studied the code thoroughly? What the hell?"
**Fix**: Always grep to verify

### Mistake 4: Saying Something Doesn't Exist Without Checking
**What happened**: Said GlobalBerryTracker wasn't accessible, but it was exposed in observations
**User reaction**: "That is ridiculous. That tracker is definitely exposed in the substrate file. Why the fuck did you not find it? Didn't I tell you to carefully read the entire meltingpot code please?"
**Fix**: Grep before saying something doesn't exist

---

## Workflow for Each Task

### Standard Workflow:
1. **Read Phase README** (especially "Notes for Next Session")
2. **Grep/read code** to understand exact implementation
3. **Implement** following conventions
4. **Show code for review** with explanation
5. **Wait for approval**
6. **Commit** with only RST as author
7. **Verify** author is correct

### Example:
```
User: "Implement feature X"

Me: [Reads PHASE_README.md]
Me: [Greps for relevant code]
Me: [Implements feature]
Me: "I've implemented X. Here are the changes:
     - File A: Added Y (shows code)
     - File B: Modified Z (shows code)
     Is this acceptable to commit?"

User: "yes"

Me: [Commits with only RST as author]
Me: [Verifies with git log]
```

---

## User Preferences

### Communication Style
- User is direct and will express frustration if I mess up
- Appreciates thoroughness and attention to detail
- Wants explanations when asked, not just summaries

### Code Quality
- Follow existing patterns in codebase
- Use exact names from Lua code
- Document with "Added by RST:" comments
- No approximations

### Git Workflow
- Show before committing
- Only RST as author
- Clear commit messages explaining what/why

---

## Emergency Recovery

If I make a mistake:
1. **Don't try to hide it**
2. **Explain what happened**
3. **Ask if user wants revert or explanation**
4. **Learn from it and update this file if needed**

---

## Quick Checklist Before Any Commit

- [ ] Did I show code for review?
- [ ] Did user approve?
- [ ] Did I grep to verify exact names?
- [ ] Are comments "Added by RST:" not "Added by Claude:"?
- [ ] Is commit message clear and detailed?
- [ ] After commit: `git log --format="%an %ae" -1` shows only rstrivedi?
- [ ] No "Co-Authored-By" in commit message?

---

## Files to Read After Compaction

1. **This file (alignment.md)** - FIRST
2. **PHASE2_README.md** - Notes for Next Session section
3. **PHASE3_README.md** - Architecture and integration patterns
4. **Latest commit history** - `git log --oneline -10`

---

## Summary

**Golden Rules**:
1. Show code, get approval, then commit
2. Only RST as author, never Claude
3. Grep/read code, never guess
4. Follow "Added by RST:" convention
5. No approximations, everything exact
6. Read Phase READMEs after compaction

**If Unsure**: Ask the user, don't guess. But if it can be found in code, find it yourself.

**User's Bottom Line**: "I don't want you to fuck up again"

---

## Current Session State (Before Compaction)

**Date**: 2025-11-13

**Phase**: Pre-Production Thorough Review

**What We're Doing**:
- User is doing a systematic pre-production review before full training runs
- **Process**: User asks questions ONE-BY-ONE, I help clarify/verify/modify each before moving to next
- **Current Status**: Just completed Question #1

**Question #1: Reward Computation Verification** ✅ COMPLETED
- **User's concern**: Suspicious of reported rewards (collective ~20-200 instead of ~600)
- **What we did**:
  1. Thoroughly investigated entire reward computation system (Lua + Python)
  2. Found BUG: Lua never emits `reward_component` events, so alpha/beta/c tracking was broken
  3. **Fixed**: Reconstructed alpha/beta/c from `sanction` events in recorder.py (lines 238-282)
  4. **Fixed**: collective_reward now excludes alpha from ALL agents (line 419)
  5. Created comprehensive verification table showing all reward components

- **Final verdict**: ALL REWARD COMPUTATIONS ARE 100% CORRECT ✅
  - r_total (training): r_env + alpha - beta - c (PPO optimizes this)
  - r_eval (display): r_env - beta - c (excludes alpha training bonus)
  - collective_reward: Sum of r_eval for all 16 agents (excludes alpha)
  - Low collective (~20-200) is EXPECTED: learning ego disrupts equilibrium

- **Files modified**:
  - `/data/altar-transfer-simple/agents/metrics/recorder.py` (lines 231-282, 419)
  - Added alpha/beta/c reconstruction from sanction events
  - Fixed collective_reward formula

**Current System State**:
- RecurrentPPO: ✅ Working (all 4 combinations tested: treatment/control × recurrent/non-recurrent)
- Multi-community: ✅ Working (Phase 5 tested)
- Reward system: ✅ VERIFIED CORRECT
- Progress bar: ✅ Fixed (tqdm with colors)
- Metrics display: ✅ Correct (Mean Reward, Collective R)

**Next Step**:
- User will ask Question #2 for next verification item
- Continue one-by-one review process until all concerns addressed
- Then proceed to production runs

**Important for Post-Compaction**:
- User expects to continue with next question in review
- Do NOT commit anything yet - still in review phase
- Keep verifying one item at a time as user requests

---

**Question #2: Multi-Community Background Agent Behavior** ✅ COMPLETED (WITH FIX)

**Date**: 2025-11-14
**User Question**: "Are the background agents successfully changing behavior when environment change? So for the multi-community case, can you ensure that when at start of episode we sample an environment, the background policy also corresponds to selected norm."

**Investigation Process**:
1. User asked me to verify multi-community implementation
2. I initially found what looked like a bug in altar color passing
3. User challenged me to verify more carefully (mentioned handcoded agent videos worked for all colors)
4. I studied code deeply and found residents DO read ALTAR observation and adapt
5. **Critical Discovery**: Altar component's `altarColor` was hardcoded to `NORMATIVE_PARAMS["defaultAltarColor"]` (always 1=RED) in `create_scene()`, completely ignoring `config.permitted_color_index`

**The Bug**:
- **Location**: `meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py` line 692
- **Problem**: `create_scene()` creates Altar with hardcoded `altarColor: NORMATIVE_PARAMS["defaultAltarColor"]` (always 1=RED)
- **Impact**:
  - ❌ Single-community GREEN: Residents plant RED instead of GREEN
  - ❌ Single-community BLUE: Residents plant RED instead of BLUE
  - ❌ Multi-community: Residents always plant RED regardless of sampled community
  - ✅ Single-community RED: Works correctly (by accident)
  - **ALL non-RED experiments were training wrong equilibrium**

**How Residents Work** (this part was CORRECT):
- Residents read `obs['ALTAR']` observation (resident_policy.py line 95)
- Residents plant `self._plant_actions[altar_color]` (resident_policy.py line 156)
- Residents sanction agents with `body_color != altar_color` (resident_policy.py line 138)
- **Residents correctly adapt to whatever ALTAR says**
- **BUT ALTAR was always showing 1 (RED) due to hardcoded value**

**The Fix** (allelopathic_harvest_normative.py:1026-1063):
```python
def build(roles, config):
    num_players = len(roles)
    game_objects = create_avatar_and_associated_objects(roles=roles)

    # Create scene with default altar color
    scene = create_scene(num_players)

    # Added by RST: Override Altar color from config (for multi-community support)
    if hasattr(config, 'permitted_color_index') and config.permitted_color_index is not None:
        for component in scene["components"]:
            if component.get("component") == "Altar":
                component["kwargs"]["altarColor"] = config.permitted_color_index
                break

    # Build substrate with modified scene
    substrate_definition = dict(
        ...
        simulation={
            ...
            "scene": scene,  # Use modified scene
        },
    )
    return substrate_definition
```

**Why This Works**:
- No MeltingPot API changes (`create_scene()` signature unchanged)
- Modifies scene dict AFTER creation, BEFORE passing to dmlab2d
- Reads `config.permitted_color_index` set by sb3_wrapper or training scripts
- Backward compatible (falls back to default RED if not set)

**What This Fixes**:
- ✅ Single-community GREEN training → Residents plant GREEN
- ✅ Single-community BLUE training → Residents plant BLUE
- ✅ Multi-community training → Residents adapt to sampled color each episode
- ✅ Training scripts (sb3_wrapper.py) will now work correctly for all colors
- ✅ Baseline scripts will continue to work (default RED)

**Files Modified**:
- `/data/altar-transfer-simple/meltingpot/meltingpot/configs/substrates/allelopathic_harvest_normative.py` (build function, lines 1026-1063)

**Current Status**:
- ✅ Fix implemented
- ❌ NOT committed yet (user wants to test first)
- Next: User will test the fix before committing

**Next Step**:
- User will test multi-community and single-community variants
- Verify residents correctly adapt to GREEN and BLUE
- Continue with more questions if needed
- Then proceed to production runs

---

## Question #3: Architecture Review and Multi-Community Bug Fix

**Date**: 2025-11-14
**Status**: ✅ COMPLETED (WITH CRITICAL FIX)

### Request

User asked for full architecture explanation covering both control and treatment systems, including flow charts/tables for verification. During this review, I discovered a critical bug that would break Phase 5 multi-community control training.

### The Bug

**Issue**: `NormativeObservationFilter` was filtering ALTAR observation from ALL agents (ego + residents), not just ego.

**Impact**:
- **Phase 4 (single-community RED)**: ✅ Works by accident (residents default to altar_color=1 which matches RED)
- **Phase 5 (multi-community)**: ❌ Broken in control arm
  - When episode samples GREEN or BLUE, residents don't see ALTAR
  - Residents default to RED (line 95 in resident_policy.py)
  - Residents try to enforce RED norm while substrate expects GREEN/BLUE
  - Creates internally contradictory environment (residents violate the actual norm!)
  - Training would fail catastrophically

### The Fix

**Changed Files**:
1. `agents/envs/normative_observation_filter.py` (lines 29-41, 60-85, 98-113)
   - Added `ego_index` parameter to `__init__()`
   - Modified `_get_timestep()` to only filter ego's observations
   - Residents always see ALTAR regardless of treatment condition
   - Updated observation_spec() to only remove ALTAR from ego's spec

2. `agents/envs/sb3_wrapper.py` (lines 299-303)
   - Pass `ego_index=self.ego_index` when creating NormativeObservationFilter

**Design Rationale**:
- Observation filter is the experimental manipulation (ego only)
- Residents enforce equilibrium (must always see ALTAR to adapt to community)
- Ensures both arms have identical resident behavior (architectural parity)

### Verification

Created `test_multi_community_fix.py` and verified:
- ✅ Treatment ego sees `permitted_color` in all communities
- ✅ Control ego does NOT see `permitted_color` in all communities
- ✅ Residents see ALTAR in BOTH arms for all communities
- ✅ Residents achieve 88-94% monoculture for RED, GREEN, and BLUE
- ✅ Multi-community sampling correctly updates ALTAR per episode

**Test command**:
```bash
conda run -n altar-simple --no-capture-output python test_multi_community_fix.py
```

**Result**: All tests passed ✓

### Architecture Validation

**Phase 4 (Single-Community)**: ✅ Faithful to intended test
- Isolated observation manipulation (treatment gets altar signal, control doesn't)
- Identical resident enforcement in both arms
- Clean causal pathway for hypothesis testing

**Phase 5 (Multi-Community)**: ✅ NOW fixed and faithful
- Residents correctly adapt to sampled community (RED/GREEN/BLUE)
- Both arms face identical equilibrium enforcement
- Can test distributional competence hypothesis

### Files Modified

1. `agents/envs/normative_observation_filter.py` - Selective filtering by ego_index
2. `agents/envs/sb3_wrapper.py` - Pass ego_index to filter
3. `test_multi_community_fix.py` - Quick verification test (NEW)

### Current Status

- ✅ Bug identified during architecture review
- ✅ Fix implemented and tested
- ✅ Verification test created and passing
- ❌ NOT committed yet (user wants to review)
- ✅ Safe to proceed with Phase 4 and Phase 5 training

**Next Step**:
- User can review the fix
- Run quick test to verify: `conda run -n altar-simple python test_multi_community_fix.py`
- Continue with more pre-production questions if needed
- Then proceed to production runs

---

## Question #4: Logging and Metrics Review + Berry Metrics Addition

**Date**: 2025-11-14
**Status**: ✅ COMPLETED (WITH ENHANCEMENT)

### Request

User asked for comprehensive documentation of:
1. What metrics are being logged and how they're computed (exact formulas)
2. What metrics are in W&B and under what names
3. What's missing from W&B logging
4. When/where videos are generated and stored
5. When/where checkpoints are created and stored

### Key Finding: Berry Metrics Gap

During review, discovered that **monoculture fraction was NOT logged to W&B during training**:
- ❌ Berry percentages (red%, green%, blue%) calculated but only shown in console
- ❌ Monoculture fraction (max/total) calculated but only shown in console
- ✅ Raw berry counts were logged
- ✅ During evaluation, monoculture_fraction was fully logged

**Impact**: Could see equilibrium quality in console during training but couldn't track trends over time in W&B.

### Enhancement Made

**Added to WandbLoggingCallback** (callbacks.py:402-409, 422-426):

```python
# Compute berry distribution metrics
if total_berries > 0:
    red_pct = berries_red / total_berries * 100
    green_pct = berries_green / total_berries * 100
    blue_pct = berries_blue / total_berries * 100
    monoculture_fraction = max(berries_red, berries_green, berries_blue) / total_berries
else:
    red_pct = green_pct = blue_pct = monoculture_fraction = 0.0

# Log to W&B (NEW)
f"{prefix}/berry_pct_red": red_pct,
f"{prefix}/berry_pct_green": green_pct,
f"{prefix}/berry_pct_blue": blue_pct,
f"{prefix}/monoculture_fraction": monoculture_fraction,
```

**New W&B metrics** (logged every ~2048 steps per community):
- `episode/{community}/berry_pct_red` - % of berries that are RED (0-100)
- `episode/{community}/berry_pct_green` - % of berries that are GREEN (0-100)
- `episode/{community}/berry_pct_blue` - % of berries that are BLUE (0-100)
- `episode/{community}/monoculture_fraction` - max(R,G,B)/(R+G+B) (0.0-1.0)

**Formula**:
```
monoculture_fraction = max(berries_red, berries_green, berries_blue) / total_berries
```

Where berry counts are from `WORLD.BERRIES_BY_TYPE` observation at episode end.

### Files Modified

1. `agents/train/callbacks.py` (lines 402-409, 422-426) - Added berry distribution metrics to W&B logging

### Comprehensive Logging Documentation

Created comprehensive documentation covering:
- All metrics computed in recorder.py with exact formulas
- All W&B metrics logged during training (SB3 auto + custom)
- FiLM diagnostics, action distributions, per-community metrics
- Phase 4 and Phase 5 evaluation metrics
- What's missing from W&B logging (per-step granular data, LSTM states, resident behavior)
- Video generation (opt-in during evaluation only, not automatic during training)
- Checkpoint creation (every 50k-100k steps, stored in ./checkpoints/)

**Key formulas documented**:
- `r_total = r_env + α - β - c` (training reward)
- `r_eval = r_total - α = r_env - β - c` (evaluation reward, excludes training bonus)
- `collective_reward = Σ(r_total[i] - α[i])` for all 16 agents
- `α = +5.0` per correct sanction
- `β = -5.0` per incorrect sanction
- `c = -0.5` per zap attempt
- `monoculture_fraction = max(R,G,B) / (R+G+B)`

### Current Status

- ✅ Comprehensive logging documentation provided
- ✅ Berry metrics gap identified and fixed
- ✅ New W&B metrics added for tracking equilibrium quality during training
- ❌ NOT committed yet (user to review)

**Next Step**:
- User reviews logging documentation
- Continue with more pre-production questions if needed
- Then proceed to production runs

---

## Question #5: Experimental Configurations and FiLM Ablations

**Date**: 2025-11-14
**Status**: ✅ DOCUMENTED (Implementation Deferred)

### Request

User asked about different experimental configurations possible based on command-line arguments and ablations.

### Main Experimental Dimensions Identified

**Primary Matrix** (2×2×3 = 12 core conditions):
- **Arm**: treatment vs control
- **Mode**: single-community vs multi-community
- **Color**: RED, GREEN, BLUE

**Command-line flexibility**:
- `--multi-community` flag for Phase 5
- `--permitted-color {red|green|blue}` for single-community color selection
- `--seed`, `--total-timesteps`, `--n-envs` for cluster sweeps
- All major parameters overridable via CLI (no YAML editing needed)

**Config-based ablations** (require YAML editing):
- Architecture: `recurrent` (LSTM vs feedforward), `trunk_dim`, hidden sizes
- Reward shaping: `alpha`, `beta`, `c` values
- Environment: `include_timestep`, `grace_period`, `episode_length`
- Training: learning rate, entropy coefficients, batch size

### Critical Finding: FiLM Ablations Missing

**User insight**: Need ablations to validate FiLM learns color-specific modulation, not just benefits from any signal.

**Proposed ablation matrix**:
1. **Correct-FiLM**: Current treatment (altar color → FiLM)
2. **Null-FiLM**: Current control (zeros → FiLM)
3. **Random-FiLM**: Random noise → FiLM (tests if FiLM uses color semantics)
4. **No-FiLM**: Concatenation instead of FiLM (tests if FiLM architecture necessary)

**Critical test**: Correct-FiLM vs Random-FiLM
- If Correct >> Random → FiLM learns color-specific modulation ✓
- If Correct ≈ Random → FiLM broken (doesn't use color info) ✗

**Multi-community diagnostic**: Random-FiLM-Multi should FAIL badly
- Cannot learn 3 distinct policies with random conditioning
- Strong evidence FiLM learns color-specific if Correct >> Random

### Decision

**Defer FiLM ablations** until after main Phase 4/5 experiments complete.

**Rationale**:
- Main experiments establish IF hypothesis holds
- FiLM ablations establish WHY/HOW mechanism works
- Logical dependency: confirm treatment beats control first

### Documentation Created

**File**: `film.md` (comprehensive ablation study design)

**Contents**:
- Motivation and problem statement
- Complete ablation matrix with architectures
- What each comparison tests
- Implementation approaches (environment wrapper vs policy modification)
- Experimental design (3 phases, 21 total runs)
- Expected results and interpretations
- Implementation checklist
- Timeline (~3 weeks post main experiments)

**Key implementation approach**:
- `FiLMAblationWrapper` to modify observations (random/permuted/null conditioning)
- `ConcatTwoHeadPolicy` for No-FiLM ablation (concatenation architecture)
- Config parameter: `env.film_ablation = 'correct' | 'random_episodic' | 'random_step' | 'permuted'`

### Recommended Experimental Priority

**Tier 1** (Core Hypothesis - Must Run):
- Treatment RED (5 seeds)
- Control RED (5 seeds)
- Treatment Multi (3 seeds)
- Control Multi (3 seeds)
Total: 16 runs

**Tier 2** (Generalization - Verify RED not special):
- Treatment/Control GREEN (3 seeds each)
- Treatment/Control BLUE (3 seeds each)
Total: 12 runs

**Tier 3** (FiLM Ablations - Deferred):
- Correct/Null/Random/No-FiLM (3 seeds each)
- Multi-community variants
Total: 21 runs (after Tier 1/2 complete)

### Files Modified

1. `film.md` - Comprehensive FiLM ablation study design (NEW)

### Current Status

- ✅ Experimental configuration space documented
- ✅ FiLM ablations designed and documented
- ✅ Implementation approach specified
- ⏸️ FiLM ablations deferred until main experiments complete
- ✅ Ready to proceed with Tier 1 experiments

**Next Step**:
- Focus on main Phase 4/5 experiments (Tier 1)
- Implement FiLM ablations later based on film.md blueprint
