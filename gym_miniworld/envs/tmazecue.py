import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..entity import MeshEnt, ImageFrame
from gym import spaces

LOW_REWARD = -100
MID_REWARD = 10
HIGH_REWARD = 100

class TMazeCue(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(
        self,
        goal_pos=None,
        rewards={'low':5, 'mid': 10, 'high': 100},
        max_episode_steps=280,
        **kwargs
    ):
        self.goal_pos = goal_pos
        self.rewards = rewards

        super().__init__(**kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

        # sample a reward
        if self.rand.bool():
            self.latent_reward = self.rewards['low']
        else:
            self.latent_reward = self.rewards['high']

        # Add a box at a random end of the hallway
        self.cue = MeshEnt(
            mesh_name='duckie',
            height=1,
            static=False
        )
        self.deterministic_goal = Box(color='purple')
        self.stochastic_goal = Box(color='grey')

        self.place_entity(self.cue, room=room1, max_z=0, min_z=0, max_x=0, min_x=0)

        self.place_entity(self.deterministic_goal, room=room2, max_z=room2.min_z + 2)
        self.place_entity(self.stochastic_goal, room=room2, min_z=room2.max_z - 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.cue):
            if self.latent_reward == self.rewards['low']:
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

class TMazeCueLeft(TMazeCue):
    def __init__(self):
        super().__init__(goal_pos=[10, 0, -6])

class TMazeCueRight(TMazeCue):
    def __init__(self):
        super().__init__(goal_pos=[10, 0, 6])
