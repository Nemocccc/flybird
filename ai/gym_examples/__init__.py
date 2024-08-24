from gym.envs.registration import register
import gym_examples
from .envs.flybird import FlappyBirdClient
register(
    id='GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)

register(
    id='Snake-v0',
    entry_point='gym_examples.envs:SnakeEnv',
    max_episode_steps=1000,
)
