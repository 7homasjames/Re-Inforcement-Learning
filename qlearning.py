import gym
import numpy as np

def q_learning(env_name='FrozenLake-v1', num_episodes=1000, max_steps=100, learning_rate=0.8, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    # Create the environment
    env = gym.make(env_name, is_slippery=False)  # Use non-slippery version for simplicity
    
    # Initialize Q-table
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    Q = np.zeros((state_space_size, action_space_size))

    # Q-learning algorithm
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Choose action using epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Take action and observe reward and next state
            next_state, reward, done, info = env.step(action)

            # Update Q-value using the Bellman equation
            Q[state, action] = Q[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("\nTraining finished. Final Q-Table:")
    print(Q)

    env.close()

if __name__ == "__main__":
    q_learning(env_name='FrozenLake-v1', num_episodes=1000)
