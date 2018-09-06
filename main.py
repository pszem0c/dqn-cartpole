import gym
import dqnagent
import numpy as np

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = dqnagent.DQNAgent(state_size, action_size)
    
    total_episodes = 2000
    total_test_episodes = 5
    max_steps = 500

    for episode in range(total_episodes):

        state = env.reset()
        state = np.reshape(state, [1, 4])

        for step in range(max_steps):

            action = agent.act(state)
    
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
    
            agent.remember(state, action, reward, next_state, done)
    
            state = next_state
    
            if done:
                print("episode {}/{}, score: {}".format(episode, total_episodes, step))
                break

        agent.replay(32)

    for episode in range(total_test_episodes):

        state = env.reset()
        state = np.reshape(state, [1, 4])

        for step in range(max_steps):

            env.render()

            action = agent.act(state)
    
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
    
            agent.remember(state, action, reward, next_state, done)
    
            state = next_state
    
            if done:
                print("episode {}/{}, score: {}".format(episode, total_test_episodes, step))
                break

env.close()
