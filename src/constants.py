WALL = -1
FOOD = 10
NONE = 1
HEAD = -2
BODY = -3
EYES = -4
FEED = -5

NORTH = (-1, 0)
SOUTH = (1, 0)
EAST = (0, 1)
WEST = (0, -1)
ORIENTATIONS = [NORTH, EAST, SOUTH, WEST]

FORWARD = 1
BACKWARD = -1
ROTATE = 0
ACTIONS = [FORWARD, BACKWARD, ROTATE]

CELL_DICT = {'#': HEAD, '+': BODY, 'o': EYES, '0': FEED, 'e': 0}

ASCII_DICT = {WALL: '■', FOOD: '=', NONE: '.',
              HEAD: '#', BODY: '+', EYES: 'o', FEED: '0'}