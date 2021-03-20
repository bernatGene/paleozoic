import numpy as np

WALL = -1
FOOD = 10
NONE = 1
HEAD = -2
BODY = -3
EYES = -4
FEED = -5
EMPTY = 0

CELLS = [HEAD, BODY, EYES, FEED]

FOOD_VALUE = 10
INITIAL_ENERGY = 50
COST_HEAD = 1
COST_FEED = 2
COST_BODY = 0.1
COST_EYES = 2

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
DEAD = '\033[0;101m'

COLORS = [RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, DEAD]

OFF = '\033[0m'
RET_LINE = '\x1b[1A\x1b[2K'

ASCII_DICT = {WALL: '■', FOOD: '@', NONE: '.',
              HEAD: '#', BODY: '+', EYES: 'o', FEED: '0', EMPTY: ' '}


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
