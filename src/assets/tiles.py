import numpy as np
from src import constants as C

W = C.WALL
O = C.NONE

TILE0 = np.array(
    [[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]]
    , dtype=np.int8)

TILE1 = np.array(
    [[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]]
    , dtype=np.int8)

TILE2 = np.array(
    [[O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]]
    , dtype=np.int8)

TILE3 = np.array(
    [[O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, W, W, O, O, O, O, O, O, O],
     [W, W, W, W, W, W, W, W, W, O, O, O, O, O, O, O],
     [W, W, W, W, W, W, W, W, W, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
     [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]]
    , dtype=np.int8)

TILES = [TILE0, TILE1, TILE2, TILE3]
RATES = [0.25, 0.25, 0.4, 0.1]


def get_random_tile(seed):
    r = np.random.default_rng(seed)
    tile = r.choice(TILES, p=RATES)
    k = r.integers(0, 3)
    rand_tile = tile.copy()
    return np.rot90(rand_tile, k)
