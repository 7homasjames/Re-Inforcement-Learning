import gym

def run_cartpole():
    # Create the CartPole environment
    env = gym.make('CartPole-v0')

    # Number of episodes to run
    num_episodes = 5

    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        state = env.reset()
        done = False
        total_reward = 0

        print(f"Episode {episode + 1} starts")

        while not done:
            # Render the environment (optional, can slow down execution)
            env.render()

            # Take a random action
            action = env.action_space.sample()

            # Step the environment
            next_state = env.step(action)


            # Accumulate the total reward
            #total_reward += reward


            # Print state, action, reward, and done flag
            print(f"State: {state}, Action: {action}")

            # Update the state for the next step
            state = next_state

        print(f"Episode {episode + 1} finished with total reward: {total_reward}\n")

    # Close the environment
    env.close()

if __name__ == "__main__":
    run_cartpole()
