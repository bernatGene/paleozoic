import time
import numpy as np

from assets import tiles
import constants as C


class Labyrinth:
    def __init__(self, field_size=64):
        self.rng = np.random.default_rng(int(time.time()))
        self.field = np.ones((field_size + 4, field_size + 4), dtype=np.int8)
        self.field_size = field_size
        self.build_walls()
        self.food_count = 0
        self.grow_food()

    def reset_seed(self, seed=0):
        if not seed:
            seed = int(time.time())
        self.rng = np.random.default_rng(seed)

    def build_walls(self):
        fs = self.field_size
        self.field[0:2, :] = C.WALL
        self.field[:, 0:2] = C.WALL
        self.field[fs + 2:, :] = C.WALL
        self.field[:, fs + 2:] = C.WALL
        rows = range(0, self.field_size, 16)  # 16 == tile_size
        cols = range(0, self.field_size, 16)
        coordinates = [(x0 + 2, y0 + 2) for x0 in rows for y0 in cols]
        for r, c in coordinates:
            self.field[r:r + 16, c:c + 16] = tiles.get_random_tile(self.rng.integers(0, 9999))

    def grow_food(self, food_count=100):
        rows = self.rng.integers(0, self.field_size + 3, food_count)
        cols = self.rng.integers(0, self.field_size + 3, food_count)
        for r, c in zip(rows, cols):
            if self.field[r][c] == C.NONE:
                self.food_count += 1
                self.field[r][c] = C.FOOD

    def crop_at_position(self, pos, shape):
        return self.field[pos[0]:pos[0]+shape[0], pos[1]:pos[1]+shape[1]].copy()

    def valid_position(self, agent_body, agent_pos):
        rows, cols = agent_body.shape
        row, col = (agent_pos[0], agent_pos[1])
        merged = np.multiply(self.field[row:row+rows, col:col+cols], agent_body)
        # print(merged)
        check_wall = np.all(merged < 1)
        if not check_wall:
            return False, -1
        check_food = np.count_nonzero(merged == C.FOOD * C.FEED)
        return True, check_food
