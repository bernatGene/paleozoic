import numpy as np

WALL = -1
FOOD = 10
NONE = 1
HEAD = -2
BODY = -3
EYES = -4
FEED = -5
EMPTY = 0  # TODO: Difference between Empty and None? How much do we depend on them being exactly 1 and 0?

CELLS = [HEAD, BODY, EYES, FEED]

FOOD_VALUE = 10
MEAT_FACTOR = 5
INVALID_PUNISH = -5
INITIAL_ENERGY = 100
COST_HEAD = 100
COST_FEED = 10
COST_BODY = 1
COST_EYES = 30
COST_FACTOR = 0.005

CELLS_COST = [COST_HEAD, COST_FEED, COST_BODY, COST_EYES]

NORTH = (-1, 0)
SOUTH = (1, 0)
EAST = (0, 1)
WEST = (0, -1)
ORIENTATIONS = [NORTH, EAST, SOUTH, WEST]

FORWARD = 1
BACKWARD = -1
ROTATE = 0
ACTIONS = [FORWARD, BACKWARD, ROTATE]

CELL_DICT = {'#': HEAD, '+': BODY, 'o': EYES, '0': FEED, '-': EMPTY}

RED = '\033[1;91m'  # Red
GREEN = '\033[1;92m'  # Green
YELLOW = '\033[1;93m'  # Yellow
BLUE = '\033[1;94m'  # Blue
MAGENTA = '\033[1;95m'  # Purple
CYAN = '\033[1;96m'  # Cyan
OVERLAP = '\033[0;103m'  # Yellow background
DEAD = '\033[0;101m'  # Red background


COLORS = [RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN]
BACKGROUND_COLORS = [OVERLAP, DEAD]

OFF = '\033[0m'
RET_LINE = '\x1b[1A\x1b[2K'

ASCII_DICT = {WALL: '■', FOOD: '@', NONE: '.',
              HEAD: '#', BODY: '+', EYES: 'o', FEED: '0', EMPTY: ' '}


def interaction_outcome(c_a, c_b):
    """
    Given two cell types which are intersecting, returns the effect to be given to each agent.
    For now, only extra mouths can predate.
    :param c_a: Cell type of agent A, must be strictly in [CELLS] list
    :param c_b: Cell type of agent B, must be strictly in [CELLS] list
    :return: Reward or punishment to each agent respectively
    """
    if c_a == FEED and c_b == FEED:
        return -FOOD_VALUE, -FOOD_VALUE
    if c_a == FEED:
        return FOOD_VALUE * MEAT_FACTOR, - FOOD_VALUE * MEAT_FACTOR
    if c_b == FEED:
        return - FOOD_VALUE * MEAT_FACTOR, FOOD_VALUE * MEAT_FACTOR
    return -1, -1


def is_cell(c):
    if c in CELLS:
        return True
    return False


def body_coordinates_vector(body, ori):
    h = np.argwhere(body == HEAD)[0]
    n = - h[0]
    e = - body.shape[1] + h[1] + 1
    s = - body.shape[0] + h[0] + 1
    w = - h[1]
    if ori == NORTH:
        return n, w
    if ori == EAST:
        return w, s
    if ori == SOUTH:
        return s, e
    if ori == WEST:
        return e, n
