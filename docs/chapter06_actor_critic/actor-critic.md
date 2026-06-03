# 6.3 Actor-Critic 架构

前两节我们认识了[优势函数](./advantage-function) $A(s,a)$ 和 [Critic 的训练方法](./critic-training)。现在让我们把所有零件组装起来，看看 Actor 和 Critic 是如何协作的。

::: tip 本节会用到的前置知识

- [优势函数 $A(s,a) = Q(s,a) - V(s)$](./advantage-function)——"这个动作比平均好了多少"
- [TD Error $\delta = r + \gamma V(s') - V(s)$](./critic-training)——优势函数的实用估计
- [策略梯度 $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$](../chapter05_policy_gradient/reinforce)——Actor 的更新公式
- [REINFORCE 与基线](../chapter05_policy_gradient/pg-improvements)——从 $G_t$ 到 $G_t - V(s)$ 的动机
  :::

## 从 REINFORCE 到 Actor-Critic

回顾第 5 章 REINFORCE 的梯度公式（回顾：[策略梯度定理](../chapter05_policy_gradient/reinforce)）：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

$G_t$ 是完整轨迹的累积回报——这就是 REINFORCE 方差大的根源。第 5 章的[基线分析](../chapter05_policy_gradient/pg-improvements)告诉我们，减掉 $V(s)$ 可以降方差。上一节我们又发现，不需要等 episode 结束——用[TD Error](./critic-training) $\delta = r + \gamma V(s') - V(s)$ 就能替代 $G_t - V(s)$ 作为优势估计：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta$$

这一替换带来的改变是根本性的：

|          | REINFORCE                 | Actor-Critic                                           |
| -------- | ------------------------- | ------------------------------------------------------ |
| 优势估计 | $G_t$（MC，需要完整轨迹） | $\delta = r + \gamma V(s') - V(s)$（TD，走一步就更新） |
| 更新时机 | episode 结束后            | 每走一步                                               |
| 方差     | 高                        | 低                                                     |
| 偏差     | 无偏                      | 有偏（[自举](../chapter03_mdp/dp-mc-td)引入偏差）      |
| 代价     | 无                        | 需要训练 Critic                                        |

### 数值对比：同一个场景下的两种更新

设 CartPole 环境中，智能体在时刻 $t$ 处于状态 $s_t$，选择动作"向右"（$a_t = \text{right}$），之后连续交互 5 步后 episode 结束。具体轨迹如下：

| 时刻  | 状态      | 动作  | 奖励 $r$ |
| ----- | --------- | ----- | -------- |
| $t$   | $s_t$     | right | 1.0      |
| $t+1$ | $s_{t+1}$ | right | 1.0      |
| $t+2$ | $s_{t+2}$ | left  | 1.0      |
| $t+3$ | $s_{t+3}$ | right | 1.0      |
| $t+4$ | $s_{t+4}$ | right | 1.0      |

取折扣因子 $\gamma = 0.99$。

**REINFORCE 的计算。** REINFORCE 必须等 episode 结束后才能更新。它从时刻 $t$ 开始计算完整回报 $G_t$：

$$
\begin{aligned}
G_t &= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \gamma^4 r_{t+5} \\
    &= 1.0 + 0.99 \times 1.0 + 0.99^2 \times 1.0 + 0.99^3 \times 1.0 + 0.99^4 \times 1.0 \\
    &= 1.0 + 0.99 + 0.9801 + 0.9703 + 0.9606 \\
    &= 4.9010.
\end{aligned}
$$

这个 $G_t$ 就是策略梯度的权重。设当前策略 $\pi_\theta$ 在 $s_t$ 下选 right 的概率为 $\pi(\text{right}|s_t) = 0.6$，则对数概率为

$$
\log \pi(\text{right}|s_t) = \log 0.6 \approx -0.5108.
$$

策略梯度更新为

$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \cdot G_t = \nabla_\theta \log \pi(\text{right}|s_t) \times 4.9010.
$$

问题在于：换一条轨迹，$G_t$ 可能是 1.0（只走了 1 步就倒了），也可能是 10.0（走了很久）。$G_t$ 的波动直接传导到梯度上，这就是 REINFORCE 方差大的来源。

> **REINFORCE 公式符号表**
>
> | 符号                                      | 含义                                                     |
> | ----------------------------------------- | -------------------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ | 策略参数 $\theta$ 的对数概率梯度，指示参数该往哪个方向调 |
> | $G_t$                                     | 从时刻 $t$ 到 episode 结束的完整折扣回报                 |
> | $r_{t+k}$                                 | 时刻 $t+k$ 获得的即时奖励                                |
> | $\gamma$                                  | 折扣因子，控制未来奖励的衰减速度                         |

**Actor-Critic 的计算。** Actor-Critic 不等 episode 结束。假设 Critic 对当前状态和下一状态的估计为 $V(s_t) = 2.0$，$V(s_{t+1}) = 3.0$。走了一步，拿到即时奖励 $r_{t+1} = 1.0$，立刻可以计算 TD Error：

$$
\begin{aligned}
\delta &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \\
       &= 1.0 + 0.99 \times 3.0 - 2.0 \\
       &= 1.0 + 2.97 - 2.0 \\
       &= 1.97.
\end{aligned}
$$

$\delta = 1.97 > 0$，说明这一步的结果比 Critic 原本的预期要好。这个正的 TD Error 直接作为优势估计：

$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \times 1.97.
$$

用同样的 $\log \pi(\text{right}|s_t) \approx -0.5108$，梯度的数量级与 REINFORCE 的相近，但权重不再是整条轨迹的累积回报，而是一步 TD Error。$\delta$ 的波动范围远小于 $G_t$——因为它只包含一步的真实随机性，而非整条轨迹的随机性叠加。

> **Actor-Critic 公式符号表**
>
> | 符号                                      | 含义                                            |
> | ----------------------------------------- | ----------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ | 策略参数 $\theta$ 的对数概率梯度                |
> | $\delta$                                  | TD Error，作为优势 $A(s,a)$ 的一步估计          |
> | $r_{t+1}$                                 | 这一步拿到的即时奖励                            |
> | $\gamma V(s_{t+1})$                       | 折扣后的下一状态价值估计（Critic 对未来的预测） |
> | $V(s_t)$                                  | Critic 对当前状态的价值估计（用作基线）         |

两种方法的核心区别可以总结为一张对比表：

| 计算步骤       | REINFORCE                      | Actor-Critic                         |
| -------------- | ------------------------------ | ------------------------------------ |
| 更新前提       | episode 结束，拿到完整轨迹     | 走一步，拿到 $r_{t+1}$ 和 $s_{t+1}$  |
| 优势估计       | $G_t = 4.9010$（5 步累积回报） | $\delta = 1.97$（一步 TD Error）     |
| 梯度权重       | 受整条轨迹随机性影响           | 只受一步随机性影响                   |
| 需要的额外组件 | 无                             | Critic 提供 $V(s_t)$ 和 $V(s_{t+1})$ |
| 每步的计算量   | 小（无网络前向传播）           | 大（Critic 要多算一次前向）          |

## Actor-Critic 架构

把优势函数和 Critic 训练整合起来，就得到了强化学习中最经典的架构。Actor 负责选择动作，Critic 负责评估动作的好坏，两者通过优势函数 $A(s,a)$ 协作：

```
Actor-Critic 数据流

  状态 s
    │
    ├──→ Actor（策略网络）
    │      π(a|s) → 选动作 a
    │                  │
    │              执行动作 a
    │                  │
    │                  ▼
    │              环境 → 返回 r, s'
    │                  │
    ├──→ Critic（价值网络）  │
    │      V(s)  ──────────┤
    │      V(s') ──────────┤
    │                      │
    │      δ = r + γV(s') - V(s)
    │            │
    │            ▼
    │      Actor 更新：θ ← θ + α·∇log π(a|s)·δ
    │      Critic 更新：V(s) ← V(s) + α·δ
    │
    └──→ 下一步，重复以上过程
```

两个网络共享同一个输入（状态 $s$），但各做各的事：

| 网络             | 角色     | 输入     | 输出                 | 学习目标         |
| ---------------- | -------- | -------- | -------------------- | ---------------- |
| Actor（演员）    | 选择动作 | 状态 $s$ | 动作概率 $\pi(a\|s)$ | 最大化累积奖励   |
| Critic（评论家） | 评估局面 | 状态 $s$ | 价值估计 $V(s)$      | 准确预测未来回报 |

如果你仔细看 Critic 的更新规则，$V(s) \leftarrow V(s) + \alpha \cdot \delta$——这不就是第 3 章的 [TD Learning](../chapter03_mdp/dp-mc-td) 吗？**Critic 本质上就是第 3 章[价值函数 $V(s)$](../chapter03_mdp/value-bellman)的神经网络实现**，它独立地学习"每个状态值多少分"。Actor 则是[策略 $\pi(a|s)$](../chapter03_mdp/policy-objective) 的神经网络实现，它根据 Critic 提供的评估来调整自己的行为。

两个函数逼近器协同工作——Critic 帮 Actor 判断"这个动作比平均好多少"，Actor 根据判断调整策略，然后新的策略又产生新的数据让 Critic 学得更好。这就是 Actor-Critic 名字的由来。

### 一步更新的完整数值推导

下面用一个具体场景走完 Actor-Critic 的一步更新。设 CartPole 中某时刻的状态向量为 $s = [0.05,\ 0.2,\ -0.03,\ 0.1]$。当前模型参数为 $\theta$，前向传播后 Actor 和 Critic 分别输出：

| 组件   | 输出                 | 数值          |
| ------ | -------------------- | ------------- |
| Actor  | 动作概率 $\pi(a\|s)$ | $[0.7,\ 0.3]$ |
| Critic | 状态价值 $V(s)$      | $1.5$         |

其中 $\pi(\text{left}|s) = 0.7$，$\pi(\text{right}|s) = 0.3$。

**第 1 步：采样动作。** 按概率采样得到 $a = \text{right}$（第 2 个动作）。对应的对数概率：

$$
\log \pi(\text{right}|s) = \log 0.3 \approx -1.2040.
$$

**第 2 步：执行动作，获取转移。** 环境返回即时奖励 $r = 1.0$，下一状态 $s' = [0.06,\ 0.25,\ -0.01,\ 0.08]$。

**第 3 步：Critic 评估下一状态。** 将 $s'$ 输入 Critic（注意此时不计算梯度）：

$$
V(s') = 2.0.
$$

**第 4 步：计算 TD 目标与 TD Error。**

$$
\begin{aligned}
\text{TD 目标} &= r + \gamma V(s') \\
               &= 1.0 + 0.99 \times 2.0 \\
               &= 1.0 + 1.98 \\
               &= 2.98.
\end{aligned}
$$

$$
\begin{aligned}
\delta &= \text{TD 目标} - V(s) \\
       &= 2.98 - 1.5 \\
       &= 1.48.
\end{aligned}
$$

$\delta = 1.48 > 0$——这一步实际拿到的好于 Critic 的预期，说明"在 $s$ 下选 right"是一个比平均更好的选择。

**第 5 步：计算 Actor Loss。**

$$
\begin{aligned}
L_{\text{actor}} &= -\log \pi(\text{right}|s) \cdot \delta \\
                 &= -(-1.2040) \times 1.48 \\
                 &= 1.2040 \times 1.48 \\
                 &= 1.7819.
\end{aligned}
$$

注意 $\delta$ 被标记为 `.detach()`——它作为常量参与 Actor Loss，不对 Critic 反传梯度。

> **Actor Loss 公式符号表**
>
> | 符号               | 含义                                                                                       |
> | ------------------ | ------------------------------------------------------------------------------------------ |
> | $L_{\text{actor}}$ | Actor 的损失函数，对其求梯度等价于策略梯度                                                 |
> | $\log \pi(a\|s)$   | 所选动作的对数概率，$\theta$ 的可微函数                                                    |
> | $\delta$           | TD Error，作为优势估计，**不参与对 Actor 的梯度计算**                                      |
> | 负号               | 让梯度上升变为梯度下降：最小化 $-\log\pi \cdot \delta$ 等价于最大化 $\log\pi \cdot \delta$ |

**第 6 步：计算 Critic Loss。**

$$
\begin{aligned}
L_{\text{critic}} &= \delta^2 \\
                  &= 1.48^2 \\
                  &= 2.1904.
\end{aligned}
$$

这是均方误差形式——让 $V(s)$ 尽可能接近 TD 目标 $r + \gamma V(s')$。

> **Critic Loss 公式符号表**
>
> | 符号                               | 含义                                               |
> | ---------------------------------- | -------------------------------------------------- |
> | $L_{\text{critic}}$                | Critic 的损失函数，驱动 $V(s)$ 逼近 TD 目标        |
> | $\delta = r + \gamma V(s') - V(s)$ | TD Error，其中 $V(s)$ 参与 Critic 的梯度计算       |
> | $\delta^2$                         | 平方确保正负误差都产生正的损失，且大误差的惩罚更重 |

**第 7 步：总损失与反向传播。**

$$
\begin{aligned}
L_{\text{total}} &= L_{\text{actor}} + L_{\text{critic}} \\
                 &= 1.7819 + 2.1904 \\
                 &= 3.9723.
\end{aligned}
$$

反向传播时，梯度流向两路：

- **Actor 路径**：$\nabla_\theta L_{\text{actor}} = -\nabla_\theta \log \pi(\text{right}|s) \cdot 1.48$。$\delta$ 作为常量，只调节梯度的幅度和方向——$\delta > 0$ 时增大 right 的概率，$\delta < 0$ 时减小。
- **Critic 路径**：$\nabla_\theta L_{\text{critic}} = 2\delta \cdot \nabla_\theta V(s) = 2 \times 1.48 \cdot \nabla_\theta V(s) = 2.96 \cdot \nabla_\theta V(s)$。$V(s)$ 是 $\theta$ 的可微函数，梯度直接调整 Critic 的预测使其更接近 TD 目标。

一步更新的完整计算链如下：

| 步骤 | 输入               | 计算                                      | 输出                       |
| ---- | ------------------ | ----------------------------------------- | -------------------------- |
| 前向 | $s$                | $\text{Actor}(s),\ \text{Critic}(s)$      | $\pi=[0.7,0.3],\ V(s)=1.5$ |
| 采样 | $\pi$              | $\text{Categorical}(\pi).\text{sample}()$ | $a=\text{right}$           |
| 环境 | $s,\ a$            | $\text{env.step}(a)$                      | $r=1.0,\ s'$               |
| 评估 | $s'$               | $\text{Critic}(s')$                       | $V(s')=2.0$                |
| TD   | $r,\ V(s'),\ V(s)$ | $r+\gamma V(s')-V(s)$                     | $\delta=1.48$              |
| 损失 | $\log\pi,\ \delta$ | $-\log\pi\cdot\delta + \delta^2$          | $L=3.9723$                 |

### 用 PyTorch 实现 Actor-Critic

Actor-Critic 的代码比 REINFORCE 多了一个 Critic 网络，但结构依然清晰：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# ==========================================
# 1. Actor-Critic 网络（共享特征提取层）
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享的特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor 头：输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 头：输出状态价值
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# ==========================================
# 2. 训练循环（每步更新，不需要等 episode 结束）
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

        # Actor 选动作，Critic 评估状态
        probs, value = model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        # Critic 评估下一个状态
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
            next_value = 0 if done else next_value

        # TD Error = 优势估计（回顾：第 6.1 节 A ≈ δ）
        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Actor 损失：策略梯度 × 优势
        actor_loss = -log_prob * td_error.detach()

        # Critic 损失：让 V(s) 接近 TD Target（回顾：第 6.2 节 L = δ²）
        critic_loss = td_error.pow(2)

        # 总损失
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

和第 5 章的 REINFORCE 代码相比，关键区别是：多了一个 Critic 网络（输出 $V(s)$），用 TD Error（`td_target - value`）替代了 $G_t$，Critic 有自己的损失函数（MSE），而且不需要跑完 episode 才更新。

### 代码的数值追踪：一个完整的训练步

下面假设模型处于某个训练阶段，追踪一次完整的循环。设当前状态 $s = [0.1,\ 0.2,\ -0.3,\ 0.4]$，折扣因子 $\gamma = 0.99$。

**前向传播。** 将 `state_t = torch.FloatTensor([0.1, 0.2, -0.3, 0.4])` 输入模型：

```python
probs, value = model(state_t)
# probs = tensor([0.6000, 0.4000])   ← Actor 输出：left 概率 0.6, right 概率 0.4
# value = tensor(1.2000)             ← Critic 输出：V(s) = 1.2
```

**采样动作与对数概率。**

```python
dist = torch.distributions.Categorical(probs)
action = dist.sample()           # action = tensor(1)，即 right
log_prob = dist.log_prob(action) # log_prob = log(0.4) = tensor(-0.9163)
```

$\log \pi(\text{right}|s) = \log 0.4 \approx -0.9163$。

**环境交互。** 执行 `action.item() = 1`（right）：

```python
next_state, reward, terminated, truncated, _ = env.step(action.item())
# reward = 1.0
# terminated = False, truncated = False
```

**评估下一状态。**

```python
with torch.no_grad():
    _, next_value = model(torch.FloatTensor(next_state))
    # next_value = tensor(2.0000)    ← V(s') = 2.0
    # done = False, 所以 next_value 不被置零
```

**计算 TD 目标与 TD Error。**

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

**计算两个损失。**

Actor Loss（$\delta$ 被 `.detach()` 切断梯度，作为常数参与计算）：

$$
L_{\text{actor}} = -\log\pi(\text{right}|s) \cdot \delta = -(-0.9163) \times 1.78 = 1.6310.
$$

```python
actor_loss = -log_prob * td_error.detach()  # = -(-0.9163) * 1.78 = tensor(1.6310)
```

Critic Loss（$\delta$ 包含 $V(s)$，梯度通过 $V(s)$ 反传到 Critic 参数）：

$$
L_{\text{critic}} = \delta^2 = 1.78^2 = 3.1684.
$$

```python
critic_loss = td_error.pow(2)  # = 1.78^2 = tensor(3.1684)
```

**总损失。**

$$
L = L_{\text{actor}} + L_{\text{critic}} = 1.6310 + 3.1684 = 4.7994.
$$

```python
loss = actor_loss + critic_loss  # = tensor(4.7994)
```

**反向传播与参数更新。** `loss.backward()` 计算梯度后，`optimizer.step()` 按学习率 $\alpha = 0.001$ 更新参数。这次更新的效果：

- **Actor 方向**：$\delta = 1.78 > 0$，说明选 right 比预期好。梯度上升会增大 $\pi(\text{right}|s)$——下次遇到类似状态时更倾向选 right。
- **Critic 方向**：$V(s) = 1.2$ 低于 TD 目标 $2.98$。$\delta^2$ 的梯度会拉高 $V(s)$，使其更接近 $r + \gamma V(s')$。

整个计算链的关键数值汇总：

| 变量          | 值         | 含义                                   |
| ------------- | ---------- | -------------------------------------- |
| `probs`       | [0.6, 0.4] | Actor 对两个动作的概率分布             |
| `value`       | 1.2        | Critic 对当前状态的估计                |
| `log_prob`    | -0.9163    | 所选动作 right 的对数概率              |
| `reward`      | 1.0        | 环境返回的即时奖励                     |
| `next_value`  | 2.0        | Critic 对下一状态的估计                |
| `td_target`   | 2.98       | $r + \gamma V(s')$                     |
| `td_error`    | 1.78       | $\delta = \text{td\textunderscore{}target} - V(s)$ |
| `actor_loss`  | 1.6310     | $-\log\pi \cdot \delta$（.detach 后）  |
| `critic_loss` | 3.1684     | $\delta^2$                             |
| `loss`        | 4.7994     | $L_{\text{actor}} + L_{\text{critic}}$ |

### CartPole 上的 Actor-Critic 训练曲线

```
Actor-Critic 在 CartPole 上的训练曲线

 500 ┤
     │                              ━━━━━━━━━━━━━━━
 400 ┤                         ━━━━
     │                    ━━━━
 300 ┤              ━━━━━
     │         ━━━━
 200 ┤    ━━━━
     │ ━━
 100 ┤╱
     └────────────────────────────────────────────
     0    50   100  150  200  250  300  350  400  450  500
                    Episode

 对比 REINFORCE 的典型曲线（更多锯齿、更慢收敛）
```

Actor-Critic 在 CartPole 上通常在 200-300 个 episode 内就能稳定到 500 分（满分），而 REINFORCE 可能需要 500+ episode 且曲线锯齿明显。这就是"用偏差换方差"的收益——每一步都有更稳定的梯度信号，策略更新不再被运气牵着走。

## Actor-Critic 的后续演进

Actor-Critic 不是终点，而是一个骨架。后续章节中你会看到它的各种变体：

| 章节                                                               | 变体              | 关键改进                                          |
| ------------------------------------------------------------------ | ----------------- | ------------------------------------------------- |
| [第 7 章 PPO](../chapter07_ppo/intro)                              | PPO-Clip          | 限制策略更新幅度，防止"步子迈太大"                |
| [第 7 章 GAE](../chapter07_ppo/gae-reward-model)                   | 广义优势估计      | 多步 TD Error 的指数加权和，精确控制偏差-方差权衡 |
| [第 9 章 DPO](../chapter09_alignment/intro)                        | 隐式 Actor-Critic | 用偏好数据替代 Critic，去掉 on-policy 的限制      |
| [第 9 章 GRPO](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) | 去掉 Critic       | 用组内均值替代 $V(s)$，省掉一个网络               |

所有的变体都共享同一个骨架：一个负责选择的网络 + 一个负责评估的信号。变化的只是"评估信号怎么来"和"选择网络怎么更新"。

<details>
<summary>思考题：既然 Actor-Critic 比 REINFORCE 好，为什么不用纯 Critic（只用 V）？</summary>

因为只有 Critic 没办法直接输出策略。Critic 学的是 $V(s)$ 或 $Q(s,a)$，从中推导策略需要用 $\arg\max_a Q(s,a)$（回顾：[贪心最优策略](../chapter03_mdp/value-q)）——但在连续动作空间中，这个 $\arg\max$ 不存在解析解（你不可能对无限多个连续值逐一比较）。

Actor 的价值在于：它直接输出动作概率，天然适用于连续动作空间。这就是为什么需要两个网络——Critic 负责"评价"，Actor 负责"选择"，缺一不可。

</details>

<details>
<summary>思考题：Actor-Critic 的"偏差"从哪来？它有害吗？</summary>

偏差来自 Critic 的[自举（Bootstrapping）](../chapter03_mdp/dp-mc-td)——Critic 用自己的估计 $V(s')$ 来更新 $V(s)$。如果 $V(s')$ 本身就不准确，误差会传播回来。这就像你用一把不准的尺子去校准另一把尺子——误差会累积。

但这种偏差不一定是坏事。适度的偏差可以换来更低的方差，整体上可能比无偏但高方差的 REINFORCE 收敛更快。第 7 章的 GAE 就是在精确控制这个"偏差-方差权衡"——用参数 $\lambda$ 在纯 TD（高偏差低方差）和纯 MC（无偏高方差）之间平滑插值。

</details>

现在让我们看看 Actor-Critic 架构在大规模应用中的表现——[Actor-Critic 的前沿大规模应用](./ac-frontier)。

---

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
