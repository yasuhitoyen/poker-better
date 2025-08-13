class GlobalLimiter():
  def __init__(self, loop_limit: int=5):
    self.loop_limit=loop_limit
    self.counter=0
  def inc(self):
    self.counter += 1
    if self.counter > self.loop_limit:
        raise Exception('Exceeded loop limit for testing')


def train_dql_agent(num_episodes:int=1000,
                    target_update_freq:int=10):
    env = texas_holdem_no_limit_v6.env()
    env.reset()

    # observation dimensions
    first_agent = env.agent_selection
    sample_obs, _, _, _, _ = env.last()

    state_dim = len(sample_obs['observation'])
    action_dim = len(sample_obs['action_mask'])
    history_dim = state_dim

    # initialize each agent
    agents = {}
    for agent_name in env.possible_agents:
        agents[agent_name] = DQLAgent(state_dim, action_dim, history_dim)

    for episode in range(num_episodes):
        env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}

        # reset all histories, starting new game
        for agent in agents.values():
            agent.history.clear()

        for agent_name in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()

            done = terminated or truncated

            if agent_name in agents:
                episode_rewards[agent_name] += reward

                if done:
                    env.step(None)
                else:
                    # legal actions
                    legal_actions = np.where(obs['action_mask'])[0].tolist()

                    # previous experience
                    agent = agents[agent_name]
                    if hasattr(agent, 'prev_state'):
                        next_state = agent.preprocess_state(obs)
                        agent.store_experience(
                            agent.prev_state, agent.prev_action, reward,
                            next_state, done, list(agent.history)
                        )

                    # select action
                    action = agent.select_action(obs, legal_actions)

                    agent.prev_state = agent.preprocess_state(obs)
                    agent.prev_action = action

                    env.step(action)
                    agent.train()
            else:
                env.step(None)

        # after a certain number of episodes, update target network
        if episode % target_update_freq == 0:
            for agent in agents.values():
                agent.update_target_network()

        if episode % 100 == 0:
            avg_rewards = {name: episode_rewards[name] for name in episode_rewards}
            print(f"Episode {episode}, Average Rewards: {avg_rewards}")

    return agents
