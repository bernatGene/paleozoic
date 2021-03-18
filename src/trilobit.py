import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import src.constants as C

GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()


class Trilobit:
    def __init__(self, dna="#", lr=0.001, num_acts=3):
        self.dna = dna
        self.dna_cost = C.COST_HEAD
        self.body = None
        self.num_actions = num_acts
        self.model = None
        self.perception_shape = None
        self.perception = None
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_memories = []
        self.critic_value_memories = []
        self.rewards_memories = []
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

    # TODO: Build models with dynamic inputs, maybe create Brain (Model) module
    def init_model(self, perception):
        perception = perception[0]   # Only single input supported
        self.energy = 10
        self.day_reward = 0
        num_actions = self.num_actions  # FORWARD, BACKWARD, ROTATE
        num_hidden = 64
        inputs = layers.Input(shape=(self.get_perception_shape()[0][1]))
        x = layers.Flatten()(inputs)
        x = layers.Dense(num_hidden, activation="relu")(x)
        common = layers.Dense(num_hidden, activation="relu")(x)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)
        self.model = keras.Model(inputs=inputs, outputs=[action, critic])
        self.build_body()
        self.perception = perception

    def save_model(self):
        self.model.save(f'model{self.dna}')

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(f"model{self.dna}", compile=False)
        except IOError as _:
            print(f"No saved model for {self.dna}")

    def reset_state(self, perception):
        self.energy = C.INITIAL_ENERGY
        self.day_reward = 0
        self.perception = perception[0]

    def act(self):
        action_probs, critic_value = self.model(np.expand_dims(self.perception, 0))
        action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
        self.action_probs_memories.append(tf.math.log(action_probs[0, action]))
        self.critic_value_memories.append(critic_value[0, 0])
        action_const = C.ACTIONS[action]
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
        else:
            reward = 1
        if perception is not None:
            self.perception = perception[0]
        else:
            reward = -10
        self.rewards_memories.append(reward)
        self.day_reward += reward
        if self.energy < 0:
            return False
        return True

    def dream(self, tape):
        self.overall_reward = 0.05 * self.day_reward + (1 - 0.05) * self.overall_reward
        returns = []
        discounted_sum = 0
        # TODO: Rethink rewards so that they reflect better desired behaviour
        for r in self.rewards_memories[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + EPS)
        returns = returns.tolist()
        history = zip(self.action_probs_memories, self.critic_value_memories, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Clear the loss and reward history
        self.action_probs_memories.clear()
        self.critic_value_memories.clear()
        self.rewards_memories.clear()

    # TODO: DNA cost does not reflect the type of cells!
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
