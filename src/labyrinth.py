import time

import numpy as np
from assets import tiles

WALL = 0
FOOD = 127
NONE = 1


class Labyrinth:
    def __init__(self, field_size=64):
        self.rng = np.random.default_rng(int(time.time()))
        self.field = np.ones((field_size + 4, field_size + 4), dtype=np.int8)
        self.field_size = field_size
        self.build_walls()
        self.grow_food()

    def reset_seed(self, seed=0):
        if not seed:
            seed = int(time.time())
        self.rng = np.random.default_rng(seed)

    def build_walls(self):
        fs = self.field_size
        self.field[0:2, :] = WALL
        self.field[:, 0:2] = WALL
        self.field[fs + 2:, :] = WALL
        self.field[:, fs + 2:] = WALL
        rows = range(0, self.field_size, 16)  # 16 == tile_size
        cols = range(0, self.field_size, 16)
        coordinates = [(x0 + 2, y0 + 2) for x0 in rows for y0 in cols]
        for r, c in coordinates:
            self.field[r:r + 16, c:c + 16] = tiles.get_random_tile(self.rng.integers(0, 9999))

    def grow_food(self, food_count=100):
        rows = self.rng.integers(0, self.field_size + 3, food_count)
        cols = self.rng.integers(0, self.field_size + 3, food_count)
        for r, c in zip(rows, cols):
            if self.field[r][c]:
                self.field[r][c] = FOOD


