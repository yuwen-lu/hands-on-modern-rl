---
title: 6.3 Actor-Critic Architecture
---

# 6.3 Actor-Critic Architecture

In the previous two sections we met the [advantage function](./advantage-function) $A(s,a)$ and the [training method for the Critic](./critic-training). Now let's assemble all the parts and see how the Actor and the Critic collaborate.

::: tip Prerequisites for This Section

- [Advantage function $A(s,a) = Q(s,a) - V(s)$](./advantage-function) -- "How much better is this action than the average?"
- [TD Error $\delta = r + \gamma V(s') - V(s)$](./critic-training) -- a practical estimate of the advantage
- [Policy gradient $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$](../chapter05_policy_gradient/reinforce) -- the Actor's update formula
- [REINFORCE and baselines](../chapter05_policy_gradient/pg-improvements) -- motivation for going from $G_t$ to $G_t - V(s)$
  :::

## From REINFORCE to Actor-Critic

Recall the gradient formula of REINFORCE from Chapter 5 (review: [policy gradient theorem](../chapter05_policy_gradient/reinforce)):

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

$G_t$ is the cumulative return over the full trajectory -- this is precisely why REINFORCE has high variance. The [baseline analysis](../chapter05_policy_gradient/pg-improvements) in Chapter 5 showed that subtracting $V(s)$ reduces variance. In the previous section we also found that we need not wait for the episode to end -- the [TD Error](./critic-training) $\delta = r + \gamma V(s') - V(s)$ can replace $G_t - V(s)$ as an advantage estimate:

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta$$

This substitution is fundamentally transformative:

|                    | REINFORCE                         | Actor-Critic                                                           |
| ------------------ | --------------------------------- | ---------------------------------------------------------------------- |
| Advantage estimate | $G_t$ (MC, needs full trajectory) | $\delta = r + \gamma V(s') - V(s)$ (TD, update after one step)         |
| Update timing      | after the episode ends            | after every step                                                       |
| Variance           | high                              | low                                                                    |
| Bias               | unbiased                          | biased (bias introduced by [bootstrapping](../chapter03_mdp/dp-mc-td)) |
| Cost               | none                              | must train a Critic                                                    |

### Numerical Comparison: Both Updates on the Same Scenario

Consider CartPole. At time step $t$ the agent is in state $s_t$, chooses action "right" ($a_t = \text{right}$), and then interacts for 5 more steps until the episode ends. The trajectory is:

| Time  | State     | Action | Reward $r$ |
| ----- | --------- | ------ | ---------- |
| $t$   | $s_t$     | right  | 1.0        |
| $t+1$ | $s_{t+1}$ | right  | 1.0        |
| $t+2$ | $s_{t+2}$ | left   | 1.0        |
| $t+3$ | $s_{t+3}$ | right  | 1.0        |
| $t+4$ | $s_{t+4}$ | right  | 1.0        |

Take discount factor $\gamma = 0.99$.

**REINFORCE computation.** REINFORCE must wait until the episode ends before updating. It computes the full return starting from time $t$:

$$
\begin{aligned}
G_t &= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \gamma^4 r_{t+5} \\
    &= 1.0 + 0.99 \times 1.0 + 0.99^2 \times 1.0 + 0.99^3 \times 1.0 + 0.99^4 \times 1.0 \\
    &= 1.0 + 0.99 + 0.9801 + 0.9703 + 0.9606 \\
    &= 4.9010.
\end{aligned}
$$

This $G_t$ serves as the weight in the policy gradient. Suppose the current policy $\pi_\theta$ assigns probability $\pi(\text{right}|s_t) = 0.6$ to the right action in state $s_t$. The log-probability is

$$
\log \pi(\text{right}|s_t) = \log 0.6 \approx -0.5108.
$$

The policy gradient update becomes

$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \cdot G_t = \nabla_\theta \log \pi(\text{right}|s_t) \times 4.9010.
$$

The problem: on a different trajectory, $G_t$ could be 1.0 (the pole fell after one step) or 10.0 (the agent survived for a long time). The fluctuation in $G_t$ propagates directly into the gradient -- this is the source of REINFORCE's high variance.

> **REINFORCE Formula Symbol Table**
>
> | Symbol                                    | Meaning                                                                                         |
> | ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ | Log-probability gradient w.r.t. policy parameters $\theta$; indicates which direction to adjust |
> | $G_t$                                     | Full discounted return from time $t$ to the end of the episode                                  |
> | $r_{t+k}$                                 | Immediate reward received at time $t+k$                                                         |
> | $\gamma$                                  | Discount factor, controlling how fast future rewards decay                                      |

**Actor-Critic computation.** Actor-Critic does not wait for the episode to end. Suppose the Critic estimates $V(s_t) = 2.0$ and $V(s_{t+1}) = 3.0$ for the current and next states. After one step, the immediate reward $r_{t+1} = 1.0$ is received, and the TD Error can be computed immediately:

$$
\begin{aligned}
\delta &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \\
       &= 1.0 + 0.99 \times 3.0 - 2.0 \\
       &= 1.0 + 2.97 - 2.0 \\
       &= 1.97.
\end{aligned}
$$

$\delta = 1.97 > 0$, meaning the outcome of this step is better than the Critic originally expected. This positive TD Error serves directly as the advantage estimate:

$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \times 1.97.
$$

Using the same $\log \pi(\text{right}|s_t) \approx -0.5108$, the gradient is of comparable magnitude to REINFORCE's, but the weight is no longer the cumulative return over the entire trajectory -- it is a single-step TD Error. The range of fluctuation in $\delta$ is far smaller than that of $G_t$, because it contains only the randomness of one step rather than the accumulated randomness of an entire trajectory.

> **Actor-Critic Formula Symbol Table**
>
> | Symbol                                    | Meaning                                                                             |
> | ----------------------------------------- | ----------------------------------------------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ | Log-probability gradient w.r.t. policy parameters $\theta$                          |
> | $\delta$                                  | TD Error, serving as a one-step estimate of the advantage $A(s,a)$                  |
> | $r_{t+1}$                                 | Immediate reward received in this step                                              |
> | $\gamma V(s_{t+1})$                       | Discounted value estimate of the next state (the Critic's prediction of the future) |
> | $V(s_t)$                                  | Critic's value estimate for the current state (used as the baseline)                |

The core differences between the two methods can be summarized in a single comparison table:

| Computation step             | REINFORCE                                       | Actor-Critic                                     |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| Update precondition          | episode ends, full trajectory available         | one step taken, $r_{t+1}$ and $s_{t+1}$ obtained |
| Advantage estimate           | $G_t = 4.9010$ (5-step cumulative return)       | $\delta = 1.97$ (one-step TD Error)              |
| Gradient weight              | affected by randomness of the entire trajectory | affected by randomness of a single step only     |
| Additional components needed | none                                            | Critic providing $V(s_t)$ and $V(s_{t+1})$       |
| Per-step computation         | small (no network forward pass)                 | larger (Critic requires an extra forward pass)   |

## Actor-Critic Architecture

Integrating the advantage function with Critic training yields one of the most classic architectures in reinforcement learning. The Actor is responsible for selecting actions, the Critic for evaluating how good they are. The two collaborate through the advantage function $A(s,a)$:

```
Actor-Critic Data Flow

  state s
    |
    +--> Actor (policy network)
    |      pi(a|s) -> choose action a
    |                  |
    |              execute action a
    |                  |
    |                  v
    |            environment -> returns r, s'
    |                  |
    +--> Critic (value network)  |
    |      V(s)  ----------------+
    |      V(s') ----------------+
    |                      |
    |      delta = r + gamma*V(s') - V(s)
    |            |
    |            v
    |      Actor update:  theta <- theta + alpha * grad log pi(a|s) * delta
    |      Critic update: V(s) <- V(s) + alpha * delta
    |
    +--> next step, repeat the above process
```

Both networks share the same input (state $s$) but perform different tasks:

| Network | Role            | Input     | Output                           | Learning objective               |
| ------- | --------------- | --------- | -------------------------------- | -------------------------------- |
| Actor   | select actions  | state $s$ | action probabilities $\pi(a\|s)$ | maximize cumulative reward       |
| Critic  | evaluate states | state $s$ | value estimate $V(s)$            | predict future return accurately |

If you look carefully at the Critic's update rule, $V(s) \leftarrow V(s) + \alpha \cdot \delta$ -- isn't this exactly [TD learning](../chapter03_mdp/dp-mc-td) from Chapter 3? **The Critic is, in essence, a neural-network implementation of the [value function $V(s)$](../chapter03_mdp/value-bellman) from Chapter 3**, independently learning "how many points each state is worth." The Actor is a neural-network implementation of the [policy $\pi(a|s)$](../chapter03_mdp/policy-objective), adjusting its behavior based on the evaluation provided by the Critic.

Two function approximators work in concert -- the Critic helps the Actor judge "how much better this action is than average," the Actor adjusts its policy accordingly, and the new policy generates new data that helps the Critic learn better. This is where the name Actor-Critic comes from.

### Complete Numerical Derivation of a Single Update Step

Let us walk through one complete Actor-Critic update step with a concrete scenario. In CartPole, suppose the state vector at some time step is $s = [0.05,\ 0.2,\ -0.03,\ 0.1]$. The current model parameters are $\theta$. After a forward pass, the Actor and Critic produce:

| Component | Output                           | Value         |
| --------- | -------------------------------- | ------------- |
| Actor     | action probabilities $\pi(a\|s)$ | $[0.7,\ 0.3]$ |
| Critic    | state value $V(s)$               | $1.5$         |

Here $\pi(\text{left}|s) = 0.7$ and $\pi(\text{right}|s) = 0.3$.

**Step 1: Sample an action.** Sample from the distribution to obtain $a = \text{right}$ (the second action). The corresponding log-probability is:

$$
\log \pi(\text{right}|s) = \log 0.3 \approx -1.2040.
$$

**Step 2: Execute the action and observe the transition.** The environment returns immediate reward $r = 1.0$ and next state $s' = [0.06,\ 0.25,\ -0.01,\ 0.08]$.

**Step 3: Critic evaluates the next state.** Feed $s'$ into the Critic (note: no gradient is computed here):

$$
V(s') = 2.0.
$$

**Step 4: Compute the TD target and TD Error.**

$$
\begin{aligned}
\text{TD target} &= r + \gamma V(s') \\
                 &= 1.0 + 0.99 \times 2.0 \\
                 &= 1.0 + 1.98 \\
                 &= 2.98.
\end{aligned}
$$

$$
\begin{aligned}
\delta &= \text{TD target} - V(s) \\
       &= 2.98 - 1.5 \\
       &= 1.48.
\end{aligned}
$$

$\delta = 1.48 > 0$ -- the actual outcome of this step exceeded the Critic's expectation, indicating that "choosing right in state $s$" was a better-than-average decision.

**Step 5: Compute the Actor Loss.**

$$
\begin{aligned}
L_{\text{actor}} &= -\log \pi(\text{right}|s) \cdot \delta \\
                 &= -(-1.2040) \times 1.48 \\
                 &= 1.2040 \times 1.48 \\
                 &= 1.7819.
\end{aligned}
$$

Note that $\delta$ is marked as `.detach()` -- it participates in the Actor Loss as a constant and does not propagate gradients back through the Critic.

> **Actor Loss Formula Symbol Table**
>
> | Symbol             | Meaning                                                                                                                               |
> | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
> | $L_{\text{actor}}$ | Actor's loss function; taking its gradient is equivalent to the policy gradient                                                       |
> | $\log \pi(a\|s)$   | Log-probability of the chosen action; a differentiable function of $\theta$                                                           |
> | $\delta$           | TD Error, serving as the advantage estimate; **does not participate in gradient computation for the Actor**                           |
> | negative sign      | Converts gradient ascent into gradient descent: minimizing $-\log\pi \cdot \delta$ is equivalent to maximizing $\log\pi \cdot \delta$ |

**Step 6: Compute the Critic Loss.**

$$
\begin{aligned}
L_{\text{critic}} &= \delta^2 \\
                  &= 1.48^2 \\
                  &= 2.1904.
\end{aligned}
$$

This is the mean-squared-error form -- it drives $V(s)$ toward the TD target $r + \gamma V(s')$.

> **Critic Loss Formula Symbol Table**
>
> | Symbol                             | Meaning                                                                                                                  |
> | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
> | $L_{\text{critic}}$                | Critic's loss function, driving $V(s)$ toward the TD target                                                              |
> | $\delta = r + \gamma V(s') - V(s)$ | TD Error, where $V(s)$ participates in the Critic's gradient computation                                                 |
> | $\delta^2$                         | Squaring ensures that both positive and negative errors produce positive loss, with larger errors penalized more heavily |

**Step 7: Total Loss and Backpropagation.**

$$
\begin{aligned}
L_{\text{total}} &= L_{\text{actor}} + L_{\text{critic}} \\
                 &= 1.7819 + 2.1904 \\
                 &= 3.9723.
\end{aligned}
$$

During backpropagation, gradients flow along two paths:

- **Actor path**: $\nabla_\theta L_{\text{actor}} = -\nabla_\theta \log \pi(\text{right}|s) \cdot 1.48$. Since $\delta$ is treated as a constant, it only scales and directs the gradient -- when $\delta > 0$, the probability of right increases; when $\delta < 0$, it decreases.
- **Critic path**: $\nabla_\theta L_{\text{critic}} = 2\delta \cdot \nabla_\theta V(s) = 2 \times 1.48 \cdot \nabla_\theta V(s) = 2.96 \cdot \nabla_\theta V(s)$. Since $V(s)$ is a differentiable function of $\theta$, the gradient directly adjusts the Critic's prediction to bring it closer to the TD target.

The complete computation chain for one update step:

| Step     | Input              | Computation                               | Output                     |
| -------- | ------------------ | ----------------------------------------- | -------------------------- |
| Forward  | $s$                | $\text{Actor}(s),\ \text{Critic}(s)$      | $\pi=[0.7,0.3],\ V(s)=1.5$ |
| Sample   | $\pi$              | $\text{Categorical}(\pi).\text{sample}()$ | $a=\text{right}$           |
| Env      | $s,\ a$            | $\text{env.step}(a)$                      | $r=1.0,\ s'$               |
| Evaluate | $s'$               | $\text{Critic}(s')$                       | $V(s')=2.0$                |
| TD       | $r,\ V(s'),\ V(s)$ | $r+\gamma V(s')-V(s)$                     | $\delta=1.48$              |
| Loss     | $\log\pi,\ \delta$ | $-\log\pi\cdot\delta + \delta^2$          | $L=3.9723$                 |

### Implementing Actor-Critic in PyTorch

Compared with REINFORCE, Actor-Critic adds a Critic network, but the overall structure remains clean:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# ==========================================
# 1. Actor-Critic network (shared feature extractor)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # shared feature extraction layer
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor head: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head: outputs state value
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# ==========================================
# 2. Training loop (update every step; no need to wait for episode end)
# ==========================================
env = gym.make("CartPole-v1")
model = ActorCritic(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

reward_history = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_t = torch.FloatTensor(state)

        # Actor chooses action; Critic evaluates state
        probs, value = model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        # Critic evaluates the next state
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
            next_value = 0 if done else next_value

        # TD Error = advantage estimate (review: Section 6.1 A ~ delta)
        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Actor loss: policy gradient x advantage
        actor_loss = -log_prob * td_error.detach()

        # Critic loss: make V(s) close to TD target (review: Section 6.2 L = delta^2)
        critic_loss = td_error.pow(2)

        # Total loss
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1} | Avg Reward: {avg:.1f}")
```

Compared with the REINFORCE code in Chapter 5, the key differences are: there is an additional Critic network (outputting $V(s)$); TD Error (`td_target - value`) replaces $G_t$; the Critic has its own loss function (MSE); and updates happen every step rather than waiting for the episode to end.

### Code Trace: A Complete Training Step

Below we assume the model is at some point during training and trace through one complete loop. Let the current state be $s = [0.1,\ 0.2,\ -0.3,\ 0.4]$ with discount factor $\gamma = 0.99$.

**Forward pass.** Feed `state_t = torch.FloatTensor([0.1, 0.2, -0.3, 0.4])` into the model:

```python
probs, value = model(state_t)
# probs = tensor([0.6000, 0.4000])   <- Actor output: left prob 0.6, right prob 0.4
# value = tensor(1.2000)             <- Critic output: V(s) = 1.2
```

**Sample action and log-probability.**

```python
dist = torch.distributions.Categorical(probs)
action = dist.sample()           # action = tensor(1), i.e. right
log_prob = dist.log_prob(action) # log_prob = log(0.4) = tensor(-0.9163)
```

$\log \pi(\text{right}|s) = \log 0.4 \approx -0.9163$.

**Environment interaction.** Execute `action.item() = 1` (right):

```python
next_state, reward, terminated, truncated, _ = env.step(action.item())
# reward = 1.0
# terminated = False, truncated = False
```

**Evaluate the next state.**

```python
with torch.no_grad():
    _, next_value = model(torch.FloatTensor(next_state))
    # next_value = tensor(2.0000)    <- V(s') = 2.0
    # done = False, so next_value is not zeroed out
```

**Compute TD target and TD Error.**

$$
\text{td\_target} = r + \gamma \cdot V(s') = 1.0 + 0.99 \times 2.0 = 2.98.
$$

$$
\text{td\_error} = \text{td\_target} - V(s) = 2.98 - 1.2 = 1.78.
$$

```python
td_target = reward + gamma * next_value  # = 1.0 + 0.99 * 2.0 = tensor(2.9800)
td_error  = td_target - value            # = 2.98 - 1.2 = tensor(1.7800)
```

**Compute both losses.**

Actor Loss ($\delta$ is detached, participating as a constant):

$$
L_{\text{actor}} = -\log\pi(\text{right}|s) \cdot \delta = -(-0.9163) \times 1.78 = 1.6310.
$$

```python
actor_loss = -log_prob * td_error.detach()  # = -(-0.9163) * 1.78 = tensor(1.6310)
```

Critic Loss ($\delta$ contains $V(s)$; gradients propagate through $V(s)$ back to Critic parameters):

$$
L_{\text{critic}} = \delta^2 = 1.78^2 = 3.1684.
$$

```python
critic_loss = td_error.pow(2)  # = 1.78^2 = tensor(3.1684)
```

**Total loss.**

$$
L = L_{\text{actor}} + L_{\text{critic}} = 1.6310 + 3.1684 = 4.7994.
$$

```python
loss = actor_loss + critic_loss  # = tensor(4.7994)
```

**Backpropagation and parameter update.** After `loss.backward()` computes the gradients, `optimizer.step()` updates parameters with learning rate $\alpha = 0.001$. The effect of this update:

- **Actor direction**: $\delta = 1.78 > 0$, indicating that choosing right was better than expected. Gradient ascent increases $\pi(\text{right}|s)$ -- next time a similar state is encountered, the agent will be more inclined to choose right.
- **Critic direction**: $V(s) = 1.2$ is below the TD target of $2.98$. The gradient from $\delta^2$ pulls $V(s)$ upward, bringing it closer to $r + \gamma V(s')$.

Summary of key values across the entire computation chain:

| Variable      | Value      | Meaning                                           |
| ------------- | ---------- | ------------------------------------------------- |
| `probs`       | [0.6, 0.4] | Actor's probability distribution over two actions |
| `value`       | 1.2        | Critic's estimate of the current state            |
| `log_prob`    | -0.9163    | Log-probability of the chosen action (right)      |
| `reward`      | 1.0        | Immediate reward returned by the environment      |
| `next_value`  | 2.0        | Critic's estimate of the next state               |
| `td_target`   | 2.98       | $r + \gamma V(s')$                                |
| `td_error`    | 1.78       | $\delta = \text{td\textunderscore{}target} - V(s)$ |
| `actor_loss`  | 1.6310     | $-\log\pi \cdot \delta$ (after .detach)           |
| `critic_loss` | 3.1684     | $\delta^2$                                        |
| `loss`        | 4.7994     | $L_{\text{actor}} + L_{\text{critic}}$            |

### Actor-Critic Training Curve on CartPole

```
Training Curve of Actor-Critic on CartPole

 500 +
     |                              ===============
 400 +                         ====
     |                    ====
 300 +              =====
     |         ====
 200 +    ====
     | ==
 100 +/
     +----------------------------------------------
     0    50   100  150  200  250  300  350  400  450  500
                    Episode

 Compare with the typical curve of REINFORCE (more jagged, slower convergence)
```

On CartPole, Actor-Critic typically stabilizes at 500 points (the maximum) within 200-300 episodes, whereas REINFORCE may need 500+ episodes and exhibits a visibly jagged curve. This is the payoff of "trading bias for variance" -- every step provides a more stable gradient signal, and policy updates are no longer driven by luck.

## Further Evolution of Actor-Critic

Actor-Critic is not the destination; it is a skeleton. In later chapters you will encounter various extensions:

| Chapter                                                              | Variant                          | Key improvement                                                                                  |
| -------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| [Chapter 7 PPO](../chapter07_ppo/intro)                              | PPO-Clip                         | Limit the size of policy updates to avoid "taking steps that are too big"                        |
| [Chapter 7 GAE](../chapter07_ppo/gae-reward-model)                   | Generalized Advantage Estimation | Exponentially weighted sum of multi-step TD errors; precisely control the bias-variance tradeoff |
| [Chapter 9 DPO](../chapter09_alignment/intro)                        | Implicit Actor-Critic            | Replace the Critic with preference data; remove the on-policy constraint                         |
| [Chapter 9 GRPO](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) | Remove the Critic                | Replace $V(s)$ with an in-group mean; save one network                                           |

All variants share the same skeleton: one network responsible for choosing, plus one signal responsible for evaluating. What changes is only "where the evaluation signal comes from" and "how the selection network is updated."

<details>
<summary>Question to think about: if Actor-Critic is better than REINFORCE, why not use a pure Critic (only V)?</summary>

Because with only a Critic, there is no way to directly output a policy. The Critic learns $V(s)$ or $Q(s,a)$, and deriving a policy from it requires $\arg\max_a Q(s,a)$ (review: [greedy optimal policy](../chapter03_mdp/value-q)). But in continuous action spaces, this $\arg\max$ has no closed-form solution -- you cannot compare infinitely many continuous values one by one.

The Actor's value lies in directly outputting action probabilities, which naturally handles continuous action spaces. This is why two networks are needed -- the Critic provides "evaluation" and the Actor provides "selection." Neither can be omitted.

</details>

<details>
<summary>Question to think about: where does the "bias" in Actor-Critic come from, and is it harmful?</summary>

The bias comes from the Critic's [bootstrapping](../chapter03_mdp/dp-mc-td) -- the Critic uses its own estimate $V(s')$ to update $V(s)$. If $V(s')$ is itself inaccurate, the error propagates backward. It is like calibrating one ruler with another inaccurate ruler -- the errors accumulate.

But this bias is not necessarily harmful. A moderate amount of bias can buy much lower variance, and overall convergence may be faster than the unbiased but high-variance REINFORCE. In Chapter 7, GAE is precisely about controlling this "bias-variance tradeoff" -- using a parameter $\lambda$ to smoothly interpolate between pure TD (high bias, low variance) and pure MC (unbiased, high variance).

</details>

Now let's look at how the Actor-Critic architecture performs in large-scale applications: [Frontiers of Large-Scale Actor-Critic Applications](./ac-frontier).

---

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
