import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces


def pointInRect(point, rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

class Preference(MiniWorldEnv):

    def __init__(self, map=None, **kwargs):
        if map is not None:
            self._map = map
        else:
            self._map = ['SHHS',
                         'HHSH',
                         'SPHH',
                         'SSHH']

        super().__init__(
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Create a long rectangular room

        textures = {'S':'reward_high', 'H':'reward_low', 'P': 'reward_low'}
        self.rewards = {'S': 0, 'H': 1, 'P': 0}

        self.size = 2
        self._room = []
        self._rewards = []

        for x in range(len(self._map)):
            for y in range(len(self._map[0])):
                print(textures[self._map[x][y]])
                if self.map[x][y] == 'P':
                    agent_pos = [x, y]

                room = self.add_rect_room(
                    min_x=x*self.size, max_x=x*self.size + self.size,
                    min_z=y*self.size, max_z=y*self.size + self.size,
                    wall_height=0,
                    floor_tex=textures[self._map[x][y]]
                )
                room.type = self._map[x][y]
                self._room.append(room)

        room = self.add_rect_room(
            min_x=0, max_x=len(self._map)*self.size,
            min_z=0, max_z=len(self._map[0])*self.size,
            floor_tex=None
        )

        # Place the agent a random distance away from the goal
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            min_x=agent_pos[0],min_z=agent_pos[1],
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        x, _, z = self.agent.pos
        for i, room in enumerate(self._room):
            if x > room.min_x and x <= room.max_x and z > room.min_z and z <= room.max_z:
                reward = self.rewards[room.type]
        print(reward)

        return obs, reward, done, info
