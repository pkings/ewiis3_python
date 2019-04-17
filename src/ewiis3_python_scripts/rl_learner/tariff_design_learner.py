import gym

import ewiis3DatabaseConnector as data


def start_env():
    env = gym.make('tariff-v0')

"""for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("hello world!")"""

# df, game_id = data.load_total_grid_consumption_and_production('Bunnie_VidyutVanika_1')
# print(df.head())
# db.hello_world()
