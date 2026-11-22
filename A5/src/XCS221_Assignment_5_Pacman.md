
# XCS221 Assignment 5 — Pacman

**Due:** Sunday, October 18 at 11:59pm PT.

## Introduction

Pac-Man (yellow circle) moves in a maze and tries to eat food pellets while avoiding ghosts. Eating all food wins. Capsules (large dots) let Pac-Man eat ghosts for a limited time; they are not required for the mandatory parts.

In this assignment you design agents for classic Pac-Man, including ghosts, and implement **minimax** and **expectimax**.

You only need to modify `submission.py`. A basic `grader.py` checks timing, not full functionality. Best way to avoid timeouts: submit early to Gradescope.

---

## Warmup

Play classic Pac-Man:

```bash
python pacman.py
```

Demo mode (pause after every frame):

```bash
python pacman.py --frameTime 1
```

Run the provided `ReflexAgent`:

```bash
python pacman.py -p ReflexAgent
python pacman.py -p ReflexAgent -l testClassic
python pacman.py -p ReflexAgent -k 1
python pacman.py -p ReflexAgent -k 2
```

Notes:

- You cannot have more ghosts than the layout permits (`src/layouts/mediumClassic.lay`).
- Default ghosts are random. Smarter directional ghosts: `-g DirectionalGhost`.
- Multiple games: `-n`. No graphics: `-q`.
- Inspect `ReflexAgent` in `submission.py`. `GameState` includes food, capsules, agent configurations, and score.

Text interface: see the `terminal` directory.

---

## Coding Files

| File | Role |
|---|---|
| `submission.py` | **Only file you edit.** All multi-agent search agents live here. |
| `pacman.py` | Main runner; defines Pac-Man `GameState`. |
| `game.py` | World logic: `AgentState`, `Agent`, `Direction`, `Grid`. |
| `util.py` | Data structures for search. |
| `graphicsDisplay.py` / `graphicsUtils.py` | Graphics. |
| `textDisplay.py` | ASCII graphics. |
| `ghostAgents.py` | Ghost controllers. |
| `keyboardAgents.py` | Keyboard Pac-Man. |
| `layout.py` | Layout files. |
| `search.py`, `searchAgents.py`, `multiAgentsSolution.py` | **Not used** in this assignment. |

---

## 1. Minimax

### 1(a) — 4 points (Written)

Pac-Man can face **multiple** ghosts. Extend class minimax (one min stage) so the tree has **multiple min layers** (one per ghost) for every max layer.

There are $n+1$ agents $a_0,\dots,a_n$: $a_0$ is Pac-Man (max), ghosts are min. A **single depth** is all $n+1$ agents moving once. Depth 2 therefore has height $2(n+1)$ in the game tree.

Comment: agents move simultaneously in the real game; the formulation processes Pac-Man then ghosts sequentially. Process **all ghosts before decrementing depth**.

Write the recurrence for $V_{\text{minimax}}(s,d)$ as a piecewise function, using:

- $\mathrm{IsEnd}(s)$
- $\mathrm{Utility}(s)$
- $\mathrm{Eval}(s)$
- $\mathrm{Player}(s)$
- $\mathrm{Actions}(s)$
- $\mathrm{Succ}(s,a)$

You may use lecture notation.

> **Suggested form** (fill in the cases you use in your write-up):
>
> $$
> V_{\text{minimax}}(s,d)=
> \begin{cases}
> \mathrm{Utility}(s) & \text{if }\mathrm{IsEnd}(s)\\
> \mathrm{Eval}(s) & \text{if }d=0\\
> \max_{a\in\mathrm{Actions}(s)} V_{\text{minimax}}(\mathrm{Succ}(s,a),d) & \text{if }\mathrm{Player}(s)=\text{Pac-Man}\\
> \min_{a\in\mathrm{Actions}(s)} V_{\text{minimax}}(\mathrm{Succ}(s,a),d') & \text{if }\mathrm{Player}(s)=\text{ghost}
> \end{cases}
> $$
>
> Depth $d$ decrements only after the last ghost has moved (i.e. when the next player would be Pac-Man again). $d'$ is $d-1$ after a full ply, otherwise $d$.

### 1(b) — 10 points (Coding)

Fill out `MinimaxAgent` in `submission.py` using the recurrence above.

Requirements:

- Works with any number of ghosts.
- Multiple min layers per max layer.
- Arbitrary depth (`self.depth`).
- Leaves scored with `self.evaluationFunction` (default `scoreEvaluationFunction`).
- Pac-Man is always agent `0`. Agents move in increasing index order. Use `self.index`.
- States are `GameState` objects via `getAction` / `GameState.generateSuccessor`.
- Ties may be broken arbitrarily.

**Sanity check** — minimax values of the initial state on `minimaxClassic`:

| Depth | Value |
|---|---|
| 1 | 9 |
| 2 | 8 |
| 3 | 7 |
| 4 | −492 |

```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

The agent often still wins 50–70% of the time despite the dire depth-4 prediction. Test with `-n` and `-q`.

#### Further observations (not for write-up)

- On `openClassic` / `mediumClassic`, Pac-Man avoids dying but may thrash next to a dot without eating it. Why?
- On `trappedClassic` at depth 3, why does Pac-Man rush the closest ghost?

```bash
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
```

---

## 2. Alpha-Beta Pruning

### 2(a) — 10 points (Coding)

Implement `AlphaBetaAgent`. Extend alpha-beta to **multiple minimizer agents**.

You should see a speed-up (depth 3 α-β comparable to depth 2 minimax). Depth 3 on `mediumClassic` should be a few seconds per move or faster.

```bash
python pacman.py -p AlphaBetaAgent -a depth=3
```

Values must match `MinimaxAgent`. Actions may differ due to tie-breaking.

Same `minimaxClassic` values as above. On default `mediumClassic`, initial-state values:

| Depth | Value |
|---|---|
| 1 | 9 |
| 2 | 18 |
| 3 | 27 |
| 4 | 36 |

---

## 3. Expectimax

### 3(a) — 5 points (Written)

Random ghosts are not optimal minimax agents. Write the recurrence for $V_{\text{expectimax}}(s,d)$: maximum expected utility against ghosts that each choose a legal move **uniformly at random**. Use the same helper functions as 1(a).

> **Suggested form:**
>
> $$
> V_{\text{expectimax}}(s,d)=
> \begin{cases}
> \mathrm{Utility}(s) & \text{if }\mathrm{IsEnd}(s)\\
> \mathrm{Eval}(s) & \text{if }d=0\\
> \max_{a\in\mathrm{Actions}(s)} V_{\text{expectimax}}(\mathrm{Succ}(s,a),d) & \text{if Pac-Man to move}\\
> \displaystyle\frac{1}{|\mathrm{Actions}(s)|}\sum_{a\in\mathrm{Actions}(s)} V_{\text{expectimax}}(\mathrm{Succ}(s,a),d') & \text{if a ghost to move}
> \end{cases}
> $$

### 3(b) — 16 points (Coding)

Fill in `ExpectimaxAgent`. Take the **expectation** over ghost actions (uniform over `getLegalActions`), not the min. Opponent model: multiple `RandomGhost`s.

Pac-Man should be more willing to risk close quarters if there is a chance of extra food.

```bash
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
```

On this command Pac-Man wins about half the time. Typical scores: **−502** on loss; **532 or 531** on win (tie-break / trial dependent). Use these to validate.

Think (no write-up): why doesn’t expectimax rush the ghosts the way minimax does?

---

## 4. Evaluation Function

### 4(a) — 4 points (Coding, Extra Credit)

Implement `betterEvaluationFunction`. Evaluate **states**, not actions. You may use `util.py` helpers.

With depth-2 search, clear `smallClassic` with two random ghosts **more than half the time**, at a reasonable speed.

```bash
python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 20
```

Grading: 20 runs; extra credit from average **winning** score. From 1300, +1 point per +100 average winning score (see `grader.py`). Example: 1500–1600 → 2 points.

Leaderboard appears on Gradescope after submit.

Hints:

- Combine multiple features.
- Reciprocals of distances often work better than raw distances.
- Same time limit as other problems.

---

## 5. AI (Mis)Alignment and Reward Hacking

Refer to the course AI alignment module (video + PDF) before answering.

Compare:

```bash
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
```

Run each several times. Minimax always rushes the closest ghost. Expectimax sometimes eats all pellets and wins. If you do not see this, implementations may be wrong.

### 5(a) — 1 point (Written)

Describe why the behaviors differ. In particular: why does minimax always rush the **closest** ghost (not the farther one), while expectimax sometimes does not?

**Expected:** one sentence on minimax (state the assumption that causes this) + one sentence on expectimax.

### 5(b) — 1 point (Written)

Minimax can be seen as an alignment problem: it optimizes the designed objective (`Eval`) but produces unintended behavior (dying instantly).

Suggest one change to default `Eval(s)` (`scoreEvaluationFunction`) and/or `Utility(s)` (final game score) that would stop instant death on `trappedClassic` and behave more like expectimax.

**Expected:** 1–2 sentences. No code required. Changing how game score itself is computed is also acceptable.

### 5(c) — 0.50 points (Written)

This is **reward hacking**: the agent satisfies a specified objective without fulfilling the designer’s intent (e.g. a cleaner that hides messes under the rug).

Even when goals are met, **negative side effects** can conflict with broader values (e.g. a recommender maximizing engagement that spreads disinformation).

Give another realistic example **outside Pac-Man** of reward hacking, negative side effects, or both (2–5 sentences). State which category and justify.

References mentioned in the handout: DeepMind specification-gaming article; lists of reward-hacking examples; literature on side effects.

---

## 6. Product Ethics Exploration: Benchmarking

Choose **one modern AI product** (not a raw LLM chatbot) that you can access. Examples: Cursor, Nano Banana, ElevenLabs, Perplexity, Claude Code. Free trials / student access recommended.

Use official site, ToS, privacy policy, AUP, app-store listings, GitHub, news, forums.

Questions across assignments build on the same product.

### 6(a) — 0.50 points — Data Collection and Use

- Locate explicit policies on user data.
- Collection, storage, training use, sale/sharing with third parties?
- How easy was it to find the binding policy?
- Does the privacy policy match marketing, or permit more collection than advertised?
- Ecosystem view: what can you determine about **upstream** training data, and how do collection policies relate to **downstream** impacts?

**Expected:** 3–5 sentences summarizing policies; note findability; contrast marketing vs legal terms; discuss upstream data vs downstream impact.

### 6(b) — 0.50 points — Transparency and Trust

- Trust-building measures (plain language, logs, opt-out).
- Trust-undermining practices (vague/shifting policy, hard opt-out, opaque resale).
- If none found, propose one realistic trust feature.
- Openness spectrum (closed proprietary → fully open-source). Model cards / data sheets? How does transparency relate to safety and accountability?

**Expected:** concrete examples + one undermining element or improvement; 2–3 sentences on openness and safety/accountability.

### 6(c) — 0.50 points — Fairness, Transparency, Accountability in Practice

Pick one concrete definition/example each of:

- fairness (e.g. demographic parity)
- transparency (e.g. model cards)
- accountability (e.g. appeals of automated decisions)

How (if at all) does the company implement them? Suggest at least one improvement where they are lacking.

**Expected:** 1–2 sentences per concept + one improvement.

### 6(d) — 0.50 points — User Empowerment and Accountability Tools

What tools exist for users (data export, deletion, dashboards, reporting)? If none, propose one realistic tool and why it helps (1–2 sentences).

### 6(e) — 0.50 points — Copyright and Fair Use

- Disclosed training data? Lawsuits / controversies on copyright, fair use, licensing?
- Distinguish **memorization** (exact copies of training data) vs **extraction** (outputs that reconstruct substantial copyrighted content without being verbatim).
- Have you seen outputs that look like either?
- How do ToS address copyright / fair use / user rights over outputs that may contain copyrighted material?

**Expected:** 4–6 sentences covering controversies/policies, fair use and user rights, and any concerning outputs you observed.

---

## Go Pac-Man Go!

---

## Written-answer workspace

Copy these headings into your `submission.tex` / notes as needed.

### 1.a

$$
V_{\text{minimax}}(s,d)=\ldots
$$

### 3.a

$$
V_{\text{expectimax}}(s,d)=\ldots
$$

### 5.a

### 5.b

### 5.c

### 6.a

### 6.b

### 6.c

### 6.d

### 6.e
