import argparse
import random

import gymnasium as gym

CELLS = ("A", "B", "C", "D")
ACTIONS = ("Up", "Down", "Left", "Right")
# Each row lists the destination for Up, Down, Left, Right.
DESTINATIONS = ((0, 2, 0, 1), (1, 3, 0, 1), (0, 2, 2, 3))
ALPHA = 0.5
GAMMA = 0.9


class ChargerGrid(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(4)
        self.state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        self.state = DESTINATIONS[self.state][action]
        terminated = self.state == 3
        reward = 10 if terminated else 0
        return self.state, reward, terminated, False, {}


def choose_action(values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    best = max(values)
    return rng.choice([action for action, value in enumerate(values)
                       if value == best])


def update(q, state, action, reward, next_state, terminated):
    best_next = 0.0 if terminated else max(q[next_state])
    target = reward + GAMMA * best_next
    old = q[state][action]
    q[state][action] = old + ALPHA * (target - old)


def show_table(q, heading):
    print(f"\n{heading}")
    print(f"{'Cell':<6}" + "".join(f"{action:>8}" for action in ACTIONS))
    for cell, values in zip(CELLS[:3], q):
        print(f"{cell:<6}" + "".join(f"{value:8.2f}" for value in values))
    print(f"{'D':<6}" + f"{'-':>8}" * len(ACTIONS))


def main(episodes, epsilon):
    env = gym.wrappers.TimeLimit(ChargerGrid(), max_episode_steps=50)
    rng = random.Random(7)
    q = [[0.0] * len(ACTIONS) for _ in CELLS[:3]]
    show_table(q, "Before training")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        route = [CELLS[state]]
        total_reward = 0

        while True:
            # Replay Right, Down twice so we can check the hand calculations.
            if episode <= 2:
                action = (3, 1)[len(route) - 1]
            else:
                action = choose_action(q[state], epsilon, rng)

            next_state, reward, terminated, truncated, _ = env.step(action)
            update(q, state, action, reward, next_state, terminated)
            route.append(CELLS[next_state])
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        if episode == 1:
            assert q[0][3] == 0 and q[1][1] == 5
        if episode == 2:
            assert q[0][3] == 2.25 and q[1][1] == 7.5
        show_table(q, f"Episode {episode} | {' -> '.join(route)} | Reward {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-learning on a charger grid")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.epsilon <= 1:
        parser.error("Use at least one episode and epsilon between 0 and 1.")
    main(args.episodes, args.epsilon)
