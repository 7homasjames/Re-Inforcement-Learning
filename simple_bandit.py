import numpy as np

class SimpleBandit:
    def __init__(self, k, epsilon, true_means=None):
        """
        Initialize the multi-armed bandit.
        :param k: Number of arms
        :param epsilon: Exploration probability
        :param true_means: Actual reward probabilities for each arm (if known, for simulation purposes)
        """
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)  # Estimated rewards for each arm
        self.N = np.zeros(k)  # Number of times each arm is chosen
        self.true_means = true_means if true_means is not None else np.random.rand(k)

    def select_action(self):
        
        """Select an action using the epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.k)
        else:
            # Exploitation: choose the action with the highest estimated reward
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))

    def get_reward(self, action):
        """Simulate receiving a reward for the chosen action."""
        return np.random.binomial(1, self.true_means[action])  # Binary reward (0 or 1)

    def update_estimates(self, action, reward):
        """Update the reward estimate and action counts."""
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def run(self, iterations):
        """Run the bandit for a specified number of iterations."""
        rewards = []
        for _ in range(iterations):
            action = self.select_action()
            reward = self.get_reward(action)
            self.update_estimates(action, reward)
            rewards.append(reward)
        return rewards

# Parameters
k = 5  # Number of arms
epsilon = 0.1  # Exploration probability
iterations = 1000  # Number of iterations

# Create and run the bandit
bandit = SimpleBandit(k, epsilon)
rewards = bandit.run(iterations)

# Print results
print("True means:", bandit.true_means)
print("Estimated Q values:", bandit.Q)
print("Number of times each arm was selected:", bandit.N)

# Optional: Plotting results
import matplotlib.pyplot as plt
plt.plot(np.cumsum(rewards) / (np.arange(1, iterations + 1)), label="Average Reward")
plt.xlabel("Iterations")
plt.ylabel("Cumulative Average Reward")
plt.title("Epsilon-Greedy Bandit")
plt.legend()
plt.show()
