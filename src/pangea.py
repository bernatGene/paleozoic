import numpy as np
from labyrinth import Labyrinth

NORTH = (-1, 0)
SOUTH = (1, 0)
EAST = (0, 1)
WEST = (0, -1)
ORIENTATIONS = [NORTH, EAST, SOUTH, WEST]

FORWARD = 1
BACKWARD = -1
ROTATE = 0
ACTIONS = [FORWARD, BACKWARD, ROTATE]

WALL = 0
FOOD = 127
NONE = 1


class Pangea:
    def __init__(self, field_size=64):
        self.labyrinth = Labyrinth(field_size)
        self.field = self.labyrinth.field
        self.agent_pos = [7, 7]
        self.agent_ori = NORTH

    def perception(self):
        px = self.agent_pos[0]
        py = self.agent_pos[1]
        percep = self.field[px - 2:px + 3, py - 2:py + 3].copy()
        return percep

    def move_and_consume(self, action):
        self.agent_pos[0] += action[0]
        self.agent_pos[1] += action[1]
        food = self.field[self.agent_pos[0], self.agent_pos[1]]
        self.field[self.agent_pos[0], self.agent_pos[1]] = 0
        return self.perception(), food

    def rotate_agent(self):
        idx = ORIENTATIONS.index(self.agent_ori)
        self.agent_ori = ORIENTATIONS[(idx + 1) % 4]

    def act(self, action):
        if action == ROTATE:
            self.rotate_agent()
            return self.perception(), 0
        y = self.agent_pos[0]
        x = self.agent_pos[1]
        y += self.agent_ori[0] * action
        x += self.agent_ori[1] * action
        if self.field[y, x] == WALL:
            return self.perception(), 0
        return self.move_and_consume(action)

    def show_field(self):
        field = self.field.copy()
        field[self.agent_pos[0], self.agent_pos[1]] = -3
        plt.imshow(field)

    def get_field(self):
        field = self.field.copy()
        field[self.agent_pos[0], self.agent_pos[1]] = -3
        return field

    def reset_field(self):
        fs = self.field_size
        self.field = np.random.rand(fs + 4, fs + 4)
        self.field[self.field < 0.8] = 0
        self.field[0:2, :] = -1
        self.field[fs + 2:, :] = -1
        self.field[:, 0:2] = -1
        self.field[:, fs + 2:] = -1
        self.agent_pos = [15, 15]
        return self.perception()

