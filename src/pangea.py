import numpy as np
import tensorflow as tf
from contextlib import ExitStack

from src.labyrinth import Labyrinth
from src.trilobit import Trilobit
from src.utils import viewer
import src.constants as C


class Pangea:
    def __init__(self, field_size=(64, 160)):
        self.labyrinth = Labyrinth(field_size)
        self.agents = [Trilobit(dna="#-0+--"), Trilobit(dna="#+----0"), Trilobit(dna="#+"), Trilobit(dna="#")]
        self.agents_pos = [[None, None] for _ in self.agents]
        self.agents_ori = [C.NORTH for _ in self.agents]
        self.init_agents()
        self.day_steps = 500
        bodies = {}
        for i, a in enumerate(self.agents):
            bodies[i] = a.body
        self.viewer = viewer.Viewer(self.labyrinth.field, bodies)

    # TODO: Define _max_agents_ safe spawning regions, assign them randomly to active agents
    def init_agents(self, reset=False):
        pos = np.array([14, 81])
        for i, a in enumerate(self.agents):
            self.agents_pos[i] = pos.copy()
            if not reset:
                a.build_body()
            perception = self.perception(i)
            if not reset:
                a.init_model(perception)
            else:
                a.reset_state(perception)
            pos = pos + np.array([10, 0])

    def run_day(self, report_steps=False, report_progress=False, max_steps=0, food_limit=1000):
        if not max_steps:
            max_steps = self.day_steps
        # self.labyrinth.reset_food(food_limit=food_limit)
        self.labyrinth = Labyrinth(field_size=(64, 160))
        self.init_agents(reset=True)
        # self.agents = [Trilobit(dna="#-0+--"), Trilobit(dna="#+----0"), Trilobit(dna="#+"), Trilobit(dna="#")]
        # self.agents_pos = [[None, None] for _ in self.agents]
        # self.agents_ori = [C.NORTH for _ in self.agents]
        # self.init_agents()

        dead_at = [0 for _ in self.agents]
        with ExitStack() as stack:
            tapes = [stack.enter_context(tf.GradientTape(watch_accessed_variables=False)) for _ in self.agents]
            for i, a in enumerate(self.agents):
                tapes[i].watch(a.model.trainable_variables)
            for step in range(max_steps):
                if all(dead_at):
                    break
                for i, agent in enumerate(self.agents):
                    if dead_at[i]:
                        continue
                    action = self.agents[i].act()
                    reward, perception = self.perform_action(action, i)
                    if not self.agents[i].react(reward, perception):
                        dead_at[i] = step + 1
                if report_steps:
                    self.report_step()
            for i, a in enumerate(self.agents):
                a.dream(tapes[i])
        if report_progress:
            for i, a in enumerate(self.agents):
                steps = f"{dead_at[i]:3d}" if dead_at[i] else "All"
                print(f"A{i} survived {steps} steps (RR:{a.overall_reward:6.1f})", end=" - ")
            print("")

    def report_step(self):
        positions = {i: (p[0], p[1]) for i, p in enumerate(self.agents_pos)}
        energies = {i: a.energy for i, a in enumerate(self.agents)}
        orientations = {i: o for i, o in enumerate(self.agents_ori)}
        self.viewer.append_step(self.labyrinth.field, positions, energies, orientations)

    def perception(self, agent_idx):
        inputs = self.agents[agent_idx].get_perception_shape()
        perception = []
        agent_pos = self.agents_pos[agent_idx]
        for inp in inputs:
            inp_pos = (inp[0][0] + agent_pos[0], inp[0][1] + agent_pos[1])
            inp_shp = inp[1]
            p = self.labyrinth.perception_around_pos(inp_pos, inp_shp)
            # If agent is looking east, we need to rotate the perception field once counter-clockwise, etc.
            ori_idx = C.ORIENTATIONS.index(self.agents_ori[agent_idx])
            p = np.rot90(p, ori_idx)
            perception.append(p)
            if p.shape != (5, 5):
                print(f"Wrong perception shape!, {agent_pos}, {ori_idx}, {inp_pos}, {inp_shp}")
        return perception

    def perform_action(self, action, agent_idx):
        agent_body = self.agents[agent_idx].body
        agent_ori = self.agents_ori[agent_idx]
        if action == C.ROTATE:
            idx = C.ORIENTATIONS.index(self.agents_ori[agent_idx])
            agent_ori = C.ORIENTATIONS[(idx + 1) % 4]
        agent_pos = self.agents_pos[agent_idx] + np.array(agent_ori) * action
        valid, reward = self.labyrinth.valid_agent_position(agent_body, agent_pos, agent_ori)
        if valid:
            if agent_pos[0] >= 66 or agent_pos[1] >= 162:
                print(f"Out of bounds! {agent_body}, {agent_pos}, {agent_ori}")
            self.agents_pos[agent_idx] = agent_pos
            self.agents_ori[agent_idx] = agent_ori
            return reward, self.perception(agent_idx)
        return -1, None
