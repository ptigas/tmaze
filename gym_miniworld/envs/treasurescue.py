import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key

LOW_REWARD = -100
MID_REWARD = 10
HIGH_REWARD = 100

class TreasuresCue(MiniWorldEnv):
    """
    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.
    """

    def __init__(self, size=6, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        super().__init__(
            max_episode_steps=400,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        # sample a reward
        if self.rand.bool():
            self.latent_reward = LOW_REWARD
        else:
            self.latent_reward = HIGH_REWARD

        self.deterministic_goal = Box(color='blue')
        self.stochastic_goal = Box(color='purple')
        self.cue = Box(color='grey')

        self.place_entity(self.deterministic_goal)
        self.place_entity(self.stochastic_goal)
        self.place_entity(self.cue)

        self.place_agent()

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.cue):
            if self.latent_reward == LOW_REWARD:
                self.entities.append(ImageFrame(
                    pos=[-1, 1.35, 0],
                    dir=0,
                    width=1.8,
                    tex_name='reward_low'
                ))
            else:
                self.entities.append(ImageFrame(
                    pos=[-1, 1.35, 0],
                    dir=0,
                    width=1.8,
                    tex_name='reward_high'
                ))
            self._render_static()

        if self.near(self.deterministic_goal):
            reward += MID_REWARD
            done = True

        if self.near(self.stochastic_goal):
            reward += self.latent_reward
            done = True

        return obs, reward, done, info