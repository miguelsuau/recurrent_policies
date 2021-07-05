from agent import Agent
import numpy as np
import os
import yaml


def learn(env, agent, total_steps, total_rollout_steps, gamma, lambd):

    obs = env.reset()
    step = 0
    episode_reward = 0

    while step < total_steps:

        rollout_step = 0
        while rollout_step < total_rollout_steps:

            action, value, log_prob, hidden_memory = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            agent.add_to_memory(obs, action, reward, done, value, log_prob, hidden_memory)
            obs = new_obs
            rollout_step += 1
            step += 1
            episode_reward += np.mean(reward)
            if done[0]:
                print(episode_reward)
                episode_reward = 0
        
        agent.bootstrap(obs, total_rollout_steps, gamma, lambd)

        if agent.buffer.is_full:
            agent.update()

def read_parameters(config_file):
    dir = os.path.dirname(__file__)
    config_file = os.path.join(dir, 'configs', config_file)
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']