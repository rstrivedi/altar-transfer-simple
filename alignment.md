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
