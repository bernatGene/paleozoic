import time
import numpy as np

from src.assets import tiles
import src.constants as C


class Labyrinth:
    def __init__(self, field_size=(64, 160)):
        self.rng = np.random.default_rng(int(time.time()))
        self.field = np.ones((field_size[0] + 4, field_size[1] + 4), dtype=np.int8)
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
        self.field[fs[0] + 2:, :] = C.WALL
        self.field[:, fs[1] + 2:] = C.WALL
        rows = range(0, self.field_size[0], 16)  # 16 == tile_size
        cols = range(0, self.field_size[1], 16)
        coordinates = [(x0 + 2, y0 + 2) for x0 in rows for y0 in cols]
        for r, c in coordinates:
            if 4 <= c / 16 <= 5:
                self.field[r:r + 16, c:c + 16] = tiles.TILE0
            else:
                self.field[r:r + 16, c:c + 16] = tiles.get_random_tile(self.rng.integers(0, 9999))
        self.field[2:18, 2:18] = tiles.TILE0

    def grow_food(self, food_limit=500):
        rows = self.rng.integers(2, self.field_size[0] + 3, food_limit - self.food_count)
        cols = self.rng.integers(2, self.field_size[1] + 3, food_limit - self.food_count)
        for r, c in zip(rows, cols):
            if self.field[r][c] == C.NONE:
                self.food_count += 1
                self.field[r][c] = C.FOOD

    def reset_food(self, food_limit=500):
        self.field[self.field == C.FOOD] = C.NONE
        self.food_count = 0
        self.grow_food(food_limit=food_limit)

    def perception_around_pos(self, pos, shape):
        r = pos[0] - (shape[0] // 2)
        c = pos[1] - (shape[1] // 2)
        return self.crop_at_position((r, c), shape)

    def crop_at_position(self, pos, shape):
        return self.field[pos[0]:pos[0]+shape[0], pos[1]:pos[1]+shape[1]].copy()

    def valid_agent_position(self, agent_body, agent_pos, agent_ori):
        body = np.rot90(agent_body, -C.ORIENTATIONS.index(agent_ori))
        vr, vc = C.body_coordinates_vector(agent_body, agent_ori)
        row = agent_pos[0] + vr
        col = agent_pos[1] + vc
        if (row + body.shape[0] >= self.field.shape[0]) or (col + body.shape[1] >= self.field.shape[1]):
            return False, -1
        crop = self.crop_at_position((row, col), body.shape)
        body_mask = (body != C.EMPTY)
        wall_mask = (crop == C.WALL)
        if np.any(wall_mask & body_mask):
            return False, -1
        food_mask = (crop == C.FOOD)
        feed_mask = (body == C.FEED) | (body == C.HEAD)
        consumed_mask = food_mask & feed_mask
        consumed = np.count_nonzero(consumed_mask)
        self.food_count -= consumed
        replace = self.field[row:row+body.shape[0], col:col+body.shape[1]]
        replace[consumed_mask] = C.NONE
        return True, consumed
