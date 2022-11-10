import gym
import torch
import matplotlib.pyplot as plt

'''
Actions: 
    0   NOOP 
    1   FIRE 
    2   RIGHT
    3   LEFT
    4   RIGHTFIRE
    5   LEFTFIRE

Observations: (RGB grayscale image)
'''

env = gym.make("ALE/Pong-v5", render_mode = 'human')
obs, info = env.reset(seed=42)

for i in range(1000):
    obs, rew, tml, trunc, info = env.step(torch.randint(0, 5, (1,)).item())
    if tml or trunc:
        obs, info = env.reset()
    # print(torch.tensor(obs).size())
    # [210, 160, 3]
env.close()

# obs = torch.tensor(obs)
# plt.imshow(obs)
# plt.show()