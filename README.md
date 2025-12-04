# rl-agent
Reinforcement Learning agent built with TenserFlow and Keras-RL

# DQN Agent for CartPole-v1 (Stable-Baselines3)

This project demonstrates a basic **Deep Q-Learning (DQN)** agent trained on the classic **CartPole-v1** reinforcement learning environment using **Stable-Baselines3** and **Gymnasium**.

The goal of the agent is to learn a control policy that balances a pole on a moving cart for as long as possible.

---

## 📌 Project Highlights

- Framework: **Stable-Baselines3 (DQN)**
- Environment: **Gymnasium — CartPole-v1**
- Vectorized environment via `DummyVecEnv`
- Training on **GPU (CUDA)**
- Replay buffer + target networks
- Exploration/Exploitation (epsilon-greedy)
- Model saving & loading
- Evaluation in deterministic mode
- Support for rendering the environment

---

## 🧠 DQN Overview

DQN learns a value function **Q(s, a)** approximated by a neural network:

\[
Q(s,a) \approx Q^\*(s,a)
\]

The model is trained using the **Bellman equation** target:

\[
y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
\]

Loss:

\[
\mathcal{L} = (y - Q(s,a))^2
\]

To stabilize training, DQN uses:

- **Replay Buffer**
- **Target Network**
- **Epsilon-Greedy Exploration**

---

## 🏋️ Training

Train a model for 200,000 timesteps:

```python
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100_000,
    exploration_final_eps=0.02,
    exploration_fraction=0.2,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    verbose=1,
)

model.learn(total_timesteps=200_000)
model.save("dqn_cartpole")
env.close()
