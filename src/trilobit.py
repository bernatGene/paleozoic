import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()


class Trilobit:
    def __init__(self, dna="#", lr=0.001, num_acts=3):
        self.dna = dna
        self.num_actions = num_acts
        self.model = None
        self.perception = None
        self.learning_rate = lr
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_memories = []
        self.critic_value_memories = []
        self.rewards_memories = []
        self.energy = 10
        self.day_reward = 0
        self.overall_reward = 0

    def get_inputs(self):
        if self.dna[0] == "#":
            return [(5, 5)]
        else:
            return 0

    def init_model(self, perception):
        num_actions = self.num_actions  # FORWARD, BACKWARD, ROTATE
        num_hidden = 128
        inputs = layers.Input(shape=(self.get_inputs()[0]))
        x = layers.Flatten()(inputs)
        x = layers.Dense(num_hidden, activation="relu")(x)
        common = layers.Dense(num_hidden, activation="relu")(x)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)
        self.model = keras.Model(inputs=inputs, outputs=[action, critic])
        self.perception = perception

    def decision(self):
        action_probs, critic_value = self.model(np.expand_dims(self.perception, 0))
        action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
        self.action_probs_memories.append(tf.math.log(action_probs[0, action]))
        self.critic_value_memories.append(critic_value[0, 0])
        return action

    def react(self, food, perception):
        self.energy += food
        self.energy -= 0.2 * len(self.dna)
        self.perception = perception
        reward = self.energy
        self.rewards_memories.append(reward)
        self.day_reward += reward
        if reward < 0:
            return False
        return True

    def dream(self):
        with tf.GradientTape() as tape:
            self.overall_reward = 0.05 * self.day_reward + (1 - 0.05) * self.overall_reward
            returns = []
            discounted_sum = 0
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
