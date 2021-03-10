import numpy as np

WALL = 0
FOOD = 127
NONE = 1
CELL = -1
BODY = -2
EYES = -3
FEED = -4

ASCII_DICT = {WALL: '■', FOOD: '=', NONE: ' ',
              CELL: '±', BODY: '+', FEED: 'O'}

DEFAULT_CREATURE = np.array([FEED], dtype=np.int8)


def field_to_string(field):
    text = ""
    for row in field:
        for item in row:
            text += ASCII_DICT[item]
        text += '\n'
    return text


class Viewer:
    def __init__(self):
        self.field = None
        self.string_field = ("X" * 64 + '\n') * 64

    def register_state(self, field, creature=DEFAULT_CREATURE, pos=(7, 7)):
        ret_field = field.copy()
        ret_field[pos[0], pos[1]] = creature[0]
        self.field = ret_field
        self.string_field = field_to_string(self.field)

    def print_last_state(self):
        print(self.string_field)
