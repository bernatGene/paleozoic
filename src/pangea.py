import numpy as np
from labyrinth import Labyrinth
from trilobit import Trilobit
from utils import viewer

import tensorflow as tf

import constants as C


class Pangea:
    def __init__(self, field_size=(64, 160)):
        self.labyrinth = Labyrinth(field_size)
        self.agent = Trilobit()
        self.agent_pos = np.array([7, 7])
        self.agent_percep = (5, 5)
        self.agent_ori = C.NORTH
        self.day_steps = 500
        self.viewer = viewer.Viewer()

    def run_day(self, report_steps=False, max_steps=0):
        if not max_steps:
            max_steps = self.day_steps
        self.labyrinth.grow_food()
        with tf.GradientTape() as tape:
            self.agent.reset_state(self.perception())
            for step in range(max_steps):
                action = self.agent.act()
                prev = self.agent_pos
                reward, perception = self.perform_action(action)
                if perception is not None:
                    perception = np.rot90(perception, C.ORIENTATIONS.index(self.agent_ori))
                if not self.agent.react(reward, perception):
                    break
                if report_steps:
                    print("Run_day:", prev, action, self.agent_pos)
                    self.viewer.append_step(self.labyrinth.field, self.agent.body, self.agent_pos,
                                            self.agent.energy, action, perception)
            print(self.agent.energy, self.agent.overall_reward, "survivied:", step)
            # print(self.agent_ori)
            # print(self.agent.body)
            self.agent.dream(tape)

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
            # print(self.agent.body)
            agent_body = np.rot90(agent_body, 1)
            # print(agent_body)
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
