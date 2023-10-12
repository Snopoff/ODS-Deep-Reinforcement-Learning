import gym
import numpy as np
import time


class CrossEntropyAgent:
    def __init__(
        self, state_n, action_n, smoothing=None, laplace_lambda=0.5, policy_lambda=0.5
    ):
        self.state_n = state_n
        self.action_n = action_n
        self.smoothing = smoothing
        self.laplace_lambda = laplace_lambda
        self.policy_lambda = policy_lambda
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if self.smoothing == "laplace" or self.smoothing == "both":
                    new_model[state] += self.laplace_lambda
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        if self.smoothing == "policy" or self.smoothing == "both":
            new_model = (
                self.policy_lambda * new_model + (1 - self.policy_lambda) * self.model
            )

        self.model = new_model


def get_state(obs):
    return obs


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {"states": [], "actions": [], "rewards": []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)

        state = get_state(obs)

        if visualize:
            time.sleep(0.5)
            env.render()

        if done:
            break

    return trajectory


env = gym.make("Taxi-v3")

state_n = 500
action_n = 6

agent = CrossEntropyAgent(state_n, action_n, smoothing="policy")
q_param = 0.36994917290783336  # 0.9
iteration_n = 100
trajectory_n = 487  # 500

for iteration in range(iteration_n):
    # policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
    print("iteration:", iteration, "mean total reward:", np.mean(total_rewards))

    # policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory["rewards"])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

trajectory = get_trajectory(env, agent, max_len=100, visualize=True)
print("total reward:", sum(trajectory["rewards"]))
print("model:")
print(agent.model)
