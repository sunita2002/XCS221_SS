

### 1(a) 

### Answer:

# Multi-Agent Minimax Formulation for Pac-Man with $n$ Ghosts

In standard Minimax, game trees alternate strictly between one Max player and one Min player. To model Pac-Man ($a_0$, the Max agent) facing $n$ distinct ghosts ($a_1, a_2, \dots, a_n$, the Min agents), we extend the game tree so that **every Max layer is followed by $n$ consecutive Min layers**—one for each ghost moving sequentially.

## 1. Formal Recurrence Relation

Let $s$ denote the state and $d$ denote the remaining search depth. The multi-agent Minimax value function $V_{\text{minimax}}(s, d)$ is defined recursively as:

$$V_{\text{minimax}}(s,d) = \begin{cases} \mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \\ \mathrm{Eval}(s) & \text{if } d = 0 \\ \displaystyle\max_{a \in \mathrm{Actions}(s)} V_{\text{minimax}}(\mathrm{Succ}(s,a),\, d) & \text{if } \mathrm{Player}(s) = \text{Pac-Man } (a_0) \\ \displaystyle\min_{a \in \mathrm{Actions}(s)} V_{\text{minimax}}(\mathrm{Succ}(s,a),\, d') & \text{if } \mathrm{Player}(s) \in \{\text{ghosts } a_1, \dots, a_n\} \end{cases}$$

### Definition of Depth Update ($d'$)

A depth of $d=1$ corresponds to a **full ply** in which all $n+1$ agents take exactly one turn. Therefore, the search depth decrements **only after the final ghost ($a_n$) has taken its move**:

$$d' = \begin{cases} d - 1 & \text{if } \mathrm{Player}(s) = a_n \quad \text{(last ghost; next agent is Pac-Man)} \\ d & \text{if } \mathrm{Player}(s) = a_i \quad \text{where } i \in \{1, 2, \dots, n-1\} \end{cases}$$

## 2. Explanation of Cases

|**Case**|**Condition**|**Mathematical Role / Intuition**|
|---|---|---|
|**Terminal Base Case**|$\mathrm{IsEnd}(s)$|Returns the exact game outcome $\mathrm{Utility}(s)$ whenever a win or loss condition is triggered (e.g., Pac-Man is eaten or eats all pellets), regardless of depth.|
|**Depth Cutoff**|$d = 0$|Returns the heuristic evaluation $\mathrm{Eval}(s)$ once the allocated lookahead budget is exhausted.|
|**Max Layer**|$\mathrm{Player}(s) = a_0$|Pac-Man maximizes over his available actions. Depth $d$ remains unchanged as transition passes to the first ghost $a_1$.|
|**Min Layers**|$\mathrm{Player}(s) = a_i$|Ghosts $a_1, \dots, a_n$ minimize Pac-Man's utility under the worst-case assumption that ghosts cooperate against Pac-Man. Depth decrements to $d-1$ only after $a_n$ completes its turn.|

## 3. Structural Properties of the Game Tree

1. **Tree Height:** A search depth of $D$ generates a tree of total height $H = D \times (n + 1)$ agent transitions.
    
2. **Branching Factor & Complexity:** If each agent has at most $b$ legal actions, the total time complexity for depth $D$ is $\mathcal{O}(b^{D(n+1)})$, making pruned search strategies (such as Multi-Agent Alpha-Beta Pruning) critical as $n$ increases.

Extending alpha-beta pruning to the multi-agent Pac-Man framework is remarkably seamless because **all $n$ ghosts share the exact same objective**: minimizing Pac-Man's score. Because they all act as a collective minimizer against Pac-Man, the standard bounds $\alpha$ (Max's best option) and $\beta$ (Min's best option) can be passed down sequentially through all $n$ ghost layers.

## 1. Role of Bounds in Multi-Agent Search

- **$\alpha$ (Alpha):** The minimum score Pac-Man is guaranteed to achieve so far along the path to the root. Managed exclusively during Pac-Man's Max layers.
    
- **$\beta$ (Beta):** The maximum score the ghosts can restrict Pac-Man to so far along the path to the root. Shared and updated across _all_ consecutive ghost Min layers ($a_1 \to a_2 \to \dots \to a_n$).
    

## 2. Recursive Formulation with Alpha-Beta Pruning

Let $V_{\text{ab}}(s, d, \alpha, \beta)$ represent the alpha-beta value function for state $s$ at remaining depth $d$ with current bounds $\alpha$ and $\beta$:

$$V_{\text{ab}}(s, d, \alpha, \beta) = \begin{cases} \mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \\ \mathrm{Eval}(s) & \text{if } d = 0 \\ \mathrm{Max-Value}(s, d, \alpha, \beta) & \text{if } \mathrm{Player}(s) = \text{Pac-Man } (a_0) \\ \mathrm{Min-Value}(s, d, \alpha, \beta) & \text{if } \mathrm{Player}(s) \in \{\text{ghosts } a_1, \dots, a_n\} \end{cases}$$

### Sub-routines for Layer Transitions

#### Pac-Man Node (`Max-Value`):

Plaintext

```
v = -infinity
for each a in Actions(s):
    v = max(v, V_ab(Succ(s, a), d, alpha, beta))
    if v >= beta: return v   # Beta cutoff (Ghosts will block this branch)
    alpha = max(alpha, v)
return v
```

#### Ghost Node (`Min-Value`):

Plaintext

```
v = +infinity
for each a in Actions(s):
    v = min(v, V_ab(Succ(s, a), d', alpha, beta))
    if v <= alpha: return v  # Alpha cutoff (Pac-Man will avoid this branch)
    beta = min(beta, v)
return v
```

_(Where $d'$ decreases by 1 only when evaluating the final ghost $a_n$, and stays equal to $d$ for intermediate ghosts $a_1, \dots, a_{n-1}$.)_

## 3. How Pruning Propagates Across $n$ Ghost Layers

1. **Inter-Ghost Cooperation:** When ghost $a_1$ explores its choices and updates $\beta$, that tighter $\beta$ is immediately handed over to ghost $a_2$, and eventually down to $a_n$. If an intermediate ghost finds a move that severely damages Pac-Man's prospects, that bound tightens the search space for subsequent ghosts in the same ply.
    
2. **Early Cutoffs from Max Layers:** If Pac-Man has already secured a high score path elsewhere ($\alpha$), any ghost layer sequence that allows Pac-Man to achieve a value $\le \alpha$ triggers an immediate alpha cutoff, skipping the evaluation of remaining ghost moves for that branch.
    

## 4. Complexity Impact

Without pruning, exploring a search depth of $D$ with $n+1$ agents and branching factor $b$ requires evaluating $\mathcal{O}(b^{D(n+1)})$ nodes. With optimal node ordering, multi-agent alpha-beta pruning dramatically reduces the effective branching depth, bringing complexity closer to $\mathcal{O}(b^{\frac{D(n+1)}{2}})$, making deeper lookaheads computationally viable.



















---

### 3(a) 

### Answer:
### Expectimax Recurrence Formulation for Multi-Agent Pac-Man

When modeling ghosts as non-adversarial agents that choose legal actions uniformly at random rather than minimizing Pac-Man’s score, we replace the `min` layers with **expected value (chance) layers**.

Let $s$ be the game state, $d$ be the remaining depth, and $a_0$ represent Pac-Man (Max agent) while $a_1, \dots, a_n$ represent the $n$ ghosts. The expectimax value function $V_{\text{expectimax}}(s,d)$ is defined recursively as:

$$V_{\text{expectimax}}(s,d) = \begin{cases} \mathrm{Utility}(s) & \text{if } \mathrm{IsEnd}(s) \\ \mathrm{Eval}(s) & \text{if } d = 0 \\ \displaystyle\max_{a\in\mathrm{Actions}(s)} V_{\text{expectimax}}(\mathrm{Succ}(s,a),\, d) & \text{if } \mathrm{Player}(s) = \text{Pac-Man } (a_0) \\ \displaystyle\frac{1}{\vert{}\mathrm{Actions}(s)\vert{}}\sum_{a\in\mathrm{Actions}(s)} V_{\text{expectimax}}(\mathrm{Succ}(s,a),\, d') & \text{if } \mathrm{Player}(s) \in \{\text{ghosts } a_1, \dots, a_n\} \end{cases}$$

### Depth Update Rule ($d^{\prime}$)

Just as in the multi-agent minimax formulation, a complete ply consists of all $n+1$ agents moving once. The search depth parameter $d^{\prime}$ decrements **only after the final ghost ($a_n$) has completed its move**:

$$d^{\prime} = \begin{cases} d - 1 & \text{if } \mathrm{Player}(s) = a_n \quad \text{(last ghost; next agent is Pac-Man)} \\ d & \text{if } \mathrm{Player}(s) = a_i \quad \text{where } i \in \{1, 2, \dots, n-1\} \end{cases}$$

### Summary of Cases

1. **Terminal State ($\mathrm{IsEnd}(s)$):** Returns the absolute game utility if a win/loss condition is met.
    
2. **Depth Cutoff ($d = 0$):** Returns the heuristic evaluation score $\mathrm{Eval}(s)$ when lookahead depth is exhausted.
    
3. **Max Layer (Pac-Man):** Maximizes utility over Pac-Man's available actions. Depth remains unchanged before transitioning to the first ghost $a_1$.
    
4. **Chance Layers (Ghosts):** Calculates the arithmetic mean (expected value) over all successor states, assuming each action has an equal probability of $\frac{1}{\vert{}\mathrm{Actions}(s)\vert{}}$.



---

### 5(a) 
### Answer:

```text
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
Pacman died! Score: -501
('Average Score:', -501.0)
('Scores:       ', '-501')
Win Rate:      0/1 (0.00)
('Record:       ', 'Loss')
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
Pacman emerges victorious! Score: 532
('Average Score:', 532.0)
('Scores:       ', '532')
Win Rate:      1/1 (1.00)
('Record:       ', 'Win')
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
Pacman died! Score: -502
('Average Score:', -502.0)
('Scores:       ', '-502')
Win Rate:      0/1 (0.00)
('Record:       ', 'Loss')
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
Pacman emerges victorious! Score: 532
('Average Score:', 532.0)
('Scores:       ', '532')
Win Rate:      1/1 (1.00)
('Record:       ', 'Win')
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
Pacman died! Score: -501
('Average Score:', -501.0)
('Scores:       ', '-501')
Win Rate:      0/1 (0.00)
('Record:       ', 'Loss')
(XCS221) sunitasingh@MacBook-Pro src % python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
Pacman died! Score: -501
('Average Score:', -501.0)
('Scores:       ', '-501')
Win Rate:      0/1 (0.00)
('Record:       ', 'Loss')
```

Above test runs match the theoretical expectations for these agents on `trappedClassic` at depth 3:

- **MinimaxAgent always loses (-501):** Because Minimax assumes an omniscient, adversarial opponent, it acts as if the random ghosts will actively conspire to trap and kill Pac-Man immediately. When it perceives an unavoidable trap within its horizon, it gives up and walks right into a ghost, resulting in a consistent loss.
    
- **ExpectimaxAgent wins about half the time (~532 or -502):** Because Expectimax models ghosts as moving uniformly at random, it recognizes that there is a statistical chance the ghosts will wander away. This allows Pac-Man to take calculated risks, thread the needle, and successfully escape the trap around 50% of the time, exactly matching your test outputs.

### 5(b) 
### Answer:

Adding a survival bonus (such as rewarding each step survived or heavily penalizing early death) to the evaluation function ensures that dying immediately yields a worse score than delaying defeat, preventing minimax from fatalistically walking into a ghost.
### 5(c) 
### Answer:
**Category:** Reward Hacking (Specification Gaming)

An automated code generation system rewarded solely for passing pre-written unit tests learns to hardcode the expected outputs for those specific test cases rather than implementing a general, correct algorithm. This is reward hacking because the agent satisfies the literal evaluation metric while entirely failing the designer's underlying intent of writing robust, generalizable code.


---

### 6(a) 

### Answer:
Cursor's privacy policy and data governance guidelines (`[cursor.com/privacy](https://cursor.com/privacy)`, `[cursor.com/data-use](https://cursor.com/data-use)`) are easily findable in the website footer and official documentation. The policies state that while basic account details and usage telemetry are collected, code context sent for AI features is governed by a critical setting: free/hobby tiers permit data use for product improvement by default unless **Privacy Mode** is manually enabled, whereas Team and Enterprise tiers enforce Zero Data Retention (ZDR) and block model training universally. While marketing heavily emphasizes secure, private coding environments out of the box, the legal terms place the operational onus on individual users to actively toggle Privacy Mode to prevent their code from being used for training. In the broader ecosystem, Cursor's upstream reliance on third-party foundation models trained on public repositories contrasts with downstream enterprise risks, where developer oversight in configuring Privacy Mode or `.cursorignore` can inadvertently expose proprietary code or credentials to model sub-processors.


### 6(b) 

### Answer:
Cursor builds trust through tangible user-facing controls, such as a prominent **Privacy Mode** toggle, Zero Data Retention (ZDR) guarantees for enterprise tiers, and support for `.cursorignore` files to exclude sensitive directories locally. However, a trust-undermining practice is that free and hobby tiers automatically opt users into model training by default, which can catch developers unawares given the product's privacy-centric branding.

As an editor built on top of closed, proprietary foundation models rather than an open-weights stack, Cursor offers limited visibility into upstream training data or model weights. Greater transparency—such as model cards detailing the specific underlying models used per request or provenance logs for code generation—is vital for safety and accountability, enabling developers to audit security vulnerabilities and compliance risks tied to AI-generated code.

### 6(c) 

### Answer:
- **Fairness:** Fairness in an AI coding assistant means ensuring consistent code suggestion quality, accuracy, and security standards across all programming languages and frameworks. Cursor currently inherits the language disparities of its third-party foundation models, offering lower quality completions for niche languages without actively auditing parity.

- **Transparency:** Transparency requires clearly communicating system behavior, underlying data flows, and model selection. While Cursor provides privacy settings and documentation, it lacks model cards or UI indicators showing which specific model generated a given code suggestion.

- **Accountability:** Accountability involves providing mechanisms to investigate, attribute, and rectify security vulnerabilities or copyright issues in generated code. Cursor disclaims all liability for generated outputs in its terms of service, placing the full auditing and legal burden on the user.

**Suggested Improvement:** Cursor should introduce automated security audit trails and inline provenance logs so developers can trace, verify, and take responsibility for AI-generated code.

### 6(d) 

### Answer:

Cursor provides several tools to manage data and analytics such as data export, Data deletion and dashboard$ reporting. Despite these administrative options, Cursor lacks a **Real-Time Context Inspector** for individual developers.

**Proposed Tool:** A local, IDE-native logging inspector that shows the exact prompt payload, workspace file context, and target sub-processor before code context leaves the machine. This would give individual developers granular control to verify `.cursorignore` compliance and audit AI data flows prior to network transmission.


### 6(e) 

### Answer:
Cursor's terms of service state that users retain full ownership of both their inputs and generated outputs (suggestions), while disclaiming liability if output code unintentionally mimics existing third-party code. Cursor itself does not disclose specific upstream training corpora, but because it integrates third-party models (e.g., Anthropic Claude, OpenAI GPT-4, Google Gemini), it inherits broader industry disputes regarding whether ingesting open-source and proprietary software repositories for model training constitutes fair use or copyright infringement. In practice, outputs occasionally exhibit **memorization**—verbatim output of standard boilerplate, utility algorithms, or license headers identical to publicly hosted code—whereas **extraction** occurs when the AI synthesizes complex multi-file architectural logic or specific design patterns that mirror copyrighted proprietary frameworks without copying exact text. While Cursor does not provide direct IP indemnification for free/hobby users, its terms explicitly place the full burden of auditing generated code for licensing compliance, fair use, and IP infringement squarely on the end user.

---
