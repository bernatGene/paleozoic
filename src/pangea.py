import numpy as np
from labyrinth import Labyrinth
from trilobit import Trilobit

import constants as C


class Pangea:
    def __init__(self, field_size=64):
        self.labyrinth = Labyrinth(field_size)
        self.agent = Trilobit()
        self.agent_pos = np.array([7, 7])
        self.agent_percep = (5, 5)
        self.agent_ori = C.NORTH
        self.day_steps = 300

    def run_day(self):
        self.agent.init_model(self.perception())
        for step in range(self.day_steps):
            action = self.agent.act()
            reward, perception = self.perform_action(action)
            self.agent.react(reward, perception)
        print(self.agent.energy)
        self.agent.dream()

    def perception(self):
        coor_position = np.array([self.agent_pos[0] - self.agent_percep[0] // 2,
                                  self.agent_pos[1] - self.agent_percep[1] // 2])
        return self.labyrinth.crop_at_position(coor_position, self.agent_percep)

    def rotate_agent(self):
        idx = C.ORIENTATIONS.index(self.agent_ori)
        self.agent_ori = C.ORIENTATIONS[(idx + 1) % 4]

    def perform_action(self, action):
        agent_body = self.agent.body.copy()
        agent_ori = self.agent_ori
        if action == C.ROTATE:
            np.rot90(agent_body, action)
            idx = C.ORIENTATIONS.index(self.agent_ori)
            agent_ori = C.ORIENTATIONS[(idx + 1) % 4]
        agent_pos = self.agent_pos + np.array(self.agent_ori) * action
        valid, reward = self.labyrinth.valid_position(agent_body, agent_pos)
        if valid:
            self.agent.body = agent_body
            self.agent_pos = agent_pos
            self.agent_ori = agent_ori
            return reward, self.perception()
        return -1, None
