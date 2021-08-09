import numpy as np

from src.labyrinth import Labyrinth
from src.trilobit import Trilobit
from src.utils import viewer
import src.constants as C


def overlaps(a, b):
    """
    Return if intervals overlap.
    """
    return min(a[1], b[1]) - max(a[0], b[0]) <= 0


class Pangea:
    def __init__(self, field_size=(64, 160), agents=None, food_limit=500):
        self.labyrinth = Labyrinth(field_size, food_limit=food_limit)
        if agents is None:
            self.agents = [Trilobit(dna="#-0+--"), Trilobit(dna="#+----0"), Trilobit(dna="#+"), Trilobit(dna="#")]
        else:
            self.agents = agents
        self.agents_pos = [[0, 0] for _ in self.agents]
        self.agents_ori = [C.NORTH for _ in self.agents]
        self.init_agents()
        self.day_steps = 500
        bodies = {}
        for i, a in enumerate(self.agents):
            bodies[i] = a.body
        # TODO: viewer field needs to be reset if lab is reset. In fact, it should not be an attribute probably
        self.viewer = viewer.Viewer(self.labyrinth.field, bodies)

    # TODO: Define _max_agents_ safe spawning regions, assign them randomly to active agents
    def init_agents(self, reset=False):
        pos = np.array([14, 7])
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
        self.labyrinth = Labyrinth(field_size=(64, 160))
        self.init_agents(reset=True)
        dead_at = [0 for _ in self.agents]
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
            a.dream()
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
            # TODO: merge labyrinth perception with bodies intersection to allow interaction between agents
            # If agent is looking east, we need to rotate the perception field once counter-clockwise, etc.
            ori_idx = C.ORIENTATIONS.index(self.agents_ori[agent_idx])
            p = np.rot90(p, ori_idx)
            perception.append(p)
            if p.shape != (5, 5):
                print(f"Wrong perception shape!, {agent_pos}, {ori_idx}, {inp_pos}, {inp_shp}")
        return perception

    def perform_action(self, action, agent_idx):
        """
        Attempts to perform the action chosen by the agent, returns the resulting reward and the perception around
        the new position. If the action is valid (no running into walls), the agent is moved and the resulting reward
        is calculated and returned with the new perception. If the action is invalid, the agent is not moved and
        the adequate punishment is returned (The perception is not calculated).
        :param action: The action to be taken as defined in the Constants file
        :param agent_idx: The agents index.
        :return: The resulting reward if the agent is moved, with the new perception, or the punishment and None.
        """
        agent_body = self.agents[agent_idx].body
        agent_ori = self.agents_ori[agent_idx]
        if action == C.ROTATE:
            idx = C.ORIENTATIONS.index(self.agents_ori[agent_idx])
            agent_ori = C.ORIENTATIONS[(idx + 1) % 4]
        agent_pos = self.agents_pos[agent_idx] + np.array(agent_ori) * action
        valid, food = self.labyrinth.valid_agent_position(agent_body, agent_pos, agent_ori)
        if valid:
            if agent_pos[0] >= 66 or agent_pos[1] >= 162:  # TODO: remove hardcoded values
                print(f"Out of bounds! {agent_body}, {agent_pos}, {agent_ori}")
            self.agents_pos[agent_idx] = agent_pos
            self.agents_ori[agent_idx] = agent_ori
            reward = food * C.FOOD_VALUE + self.agent_intersections(agent_idx)
            return reward, self.perception(agent_idx)
        return C.INVALID_PUNISH, None

    def agent_intersections(self, agent_idx):
        # TODO: Untested
        """
        Given an agent (Which has just moved into a valid position), calculate the intersections with other
        agents. This method will mutate the state of the passive intersected agent, by calling its "passive_react"
        method.
        :param agent_idx:
        :return:
        """
        active = self.agents[agent_idx]
        active_ori = self.agents_ori[agent_idx]
        active_pos = self.agents_pos[agent_idx]
        largest_side = max(active.body.shape[0], active.body.shape[1])
        bbox_active = (active_pos[0] - largest_side,
                       active_pos[0] + largest_side,
                       active_pos[1] - largest_side,
                       active_pos[1] + largest_side)
        intersections = []
        for idx, passive in enumerate(self.agents):
            if idx == agent_idx:
                continue
            largest_side = max(passive.body.shape[0], passive.body.shape[1])
            bbox_passive = (self.agents_pos[idx][0] - largest_side,
                            self.agents_pos[idx][0] + largest_side,
                            self.agents_pos[idx][1] - largest_side,
                            self.agents_pos[idx][1] + largest_side)
            if not (overlaps(bbox_active[0:2], bbox_passive[0:2]) and (overlaps(bbox_active[2:], bbox_passive[2:]))):
                # Safe overlap zone to avoid too much calculation.
                continue

            body_active = np.rot90(active.body, -C.ORIENTATIONS.index(active_ori))
            vr, vc = C.body_coordinates_vector(body_active, active_ori)
            active_row = active_pos[0] + vr
            active_col = active_pos[1] + vc
            tight_bbox_a = (active_row, active_row + body_active.shape[0],
                            active_col, active_col + body_active.shape[1])

            body_passive = np.rot90(passive.body, -C.ORIENTATIONS.index(self.agents_ori[idx]))
            passive_pos = self.agents_pos[idx]
            vr, vc = C.body_coordinates_vector(body_passive, self.agents_ori[idx])
            passive_row = passive_pos[0] + vr
            passive_col = passive_pos[1] + vc
            tight_bbox_p = (passive_row, passive_row + body_passive.shape[0],
                            passive_col, passive_col + body_passive.shape[1])

            min_row = min(active_row, passive_row)
            max_row = max(tight_bbox_a[1], tight_bbox_p[1])
            min_col = min(active_col, passive_col)
            max_col = max(tight_bbox_a[3], tight_bbox_p[3])
            bbox_a = np.zeros((max_row - min_row, max_col - min_col), dtype=np.int8)
            rp_a = (active_row - min_row, active_col - min_col)
            bbox_a[rp_a[0]:rp_a[0] + body_active.shape[0], rp_a[1]:rp_a[1] + body_active.shape[1]] = body_active
            bbox_p = np.zeros((max_row - min_row, max_col - min_col), dtype=np.int8)
            rp_p = (passive_row - min_row, passive_col - min_col)
            bbox_p[rp_p[0]:rp_p[0] + body_passive.shape[0], rp_p[1]:rp_p[1] + body_passive.shape[1]] = body_passive
            intersect_mask = np.argwhere(bbox_a * bbox_p != 0)  # Note: requires empty body cells to be 0
            for x, y in intersect_mask:
                intersections.append((bbox_a[x, y], bbox_p[x, y], idx))

        for (c_active, c_passive, i_passive) in intersections:
            out_active, out_passive = C.interaction_outcome(c_active, c_passive)
            print(f'Intersection: {active.dna} with {self.agents[i_passive].dna}, {out_active}, {out_passive}')
            print(f'at: {self.agents_pos[agent_idx]}  and {self.agents_pos[i_passive]}')
            self.agents[i_passive].passive_react(out_passive)
        return 0
