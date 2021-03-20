import numpy as np

import src.constants as C
from src.brain import Brain


class Trilobit:
    def __init__(self, dna="#", num_acts=3):
        self.dna = dna
        self.dna_cost = C.COST_HEAD
        self.body = None
        self.num_actions = num_acts
        self.craneum = Brain()
        self.perception_shape = None
        self.energy = C.INITIAL_ENERGY
        self.day_reward = 0
        self.overall_reward = 0

    def get_perception_shape(self):
        if self.perception_shape is None:
            eyes_pos = np.argwhere(self.body == C.EYES)
            head_pos = np.argwhere(self.body == C.HEAD)
            inputs = [((0, 0), (5, 5))]
            for r, c in eyes_pos:
                print("Warning, multiple inputs not supported yet")
            self.perception_shape = inputs
            return inputs
        return self.perception_shape

    # TODO: Rethink if this is necessary
    def init_model(self, perception):
        self.craneum.init_perception(perception[0])

    # TODO: Transfer functionality to brain module
    # def save_model(self):
    #     self.model.save(f'model{self.dna}')
    #
    # def load_model(self):
    #     try:
    #         self.model = tf.keras.models.load_model(f"tests/model{self.dna}", compile=False)
    #     except IOError as _:
    #         print(f"No saved model for {self.dna}")

    def reset_state(self, perception):
        self.energy = C.INITIAL_ENERGY
        self.day_reward = 0
        self.craneum.init_perception(perception[0])

    def act(self):
        a_index = self.craneum.decision()
        action_const = C.ACTIONS[a_index]
        if action_const == C.ROTATE:
            self.energy -= 0.5 * self.dna_cost
        elif action_const == C.BACKWARD:
            self.energy -= 0.75 * self.dna_cost
        return action_const

    def react(self, food, perception):
        self.energy += food * C.FOOD_VALUE
        self.energy -= self.dna_cost
        if food > 0:
            reward = self.energy
        elif perception is None:
            reward = -10
        else:
            reward = 1
        self.craneum.remember_reward(reward, perception)
        self.day_reward += reward
        if self.energy < 0:
            return False
        return True

    def dream(self):
        self.overall_reward = 0.05 * self.day_reward + (1 - 0.05) * self.overall_reward
        self.craneum.dream()

    def grow_gene(self, idx_dna, cell_dict, gene_pos):
        idx_dna += 1
        if idx_dna >= len(self.dna):
            return idx_dna
        gene = self.dna[idx_dna]
        if gene == '-' or (gene_pos in cell_dict):
            return idx_dna
        cell_dict[gene_pos] = gene
        if gene == 'o' or gene == '0':
            return idx_dna
        for i in range(4):
            ori = C.ORIENTATIONS[i]
            child_gene_pos = (gene_pos[0] + ori[0], gene_pos[1] + ori[1])
            idx_dna = self.grow_gene(idx_dna, cell_dict, child_gene_pos)
        return idx_dna

    def build_body(self):
        gene_pos = (0, 0)
        cell_dict = {}
        self.grow_gene(-1, cell_dict, gene_pos)
        self.array_body(cell_dict)

    def array_body(self, cell_dict):
        max_r = max([k[0] for k in cell_dict.keys()])
        min_r = min([k[0] for k in cell_dict.keys()])
        max_c = max([k[1] for k in cell_dict.keys()])
        min_c = min([k[1] for k in cell_dict.keys()])
        rows = max_r - min_r + 1
        cols = max_c - min_c + 1
        self.body = np.zeros((rows, cols), dtype=np.int8)
        for k, v in cell_dict.items():
            r = k[0] - min_r
            c = k[1] - min_c
            self.body[r, c] = C.CELL_DICT[v]
        for i, cell_type in enumerate(C.CELLS):
            n_cells = np.count_nonzero(self.body == cell_type)
            self.dna_cost += n_cells * C.CELLS_COST[i]

