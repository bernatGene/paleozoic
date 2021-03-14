import numpy as np
from src import constants as C

DEFAULT_CREATURE = np.array([C.HEAD], dtype=np.int8)


def field_to_string(field):
    text = ""
    for row in field:
        for item in row:
            text += C.ASCII_DICT[item]
        text += '\n'
    return text


class Viewer:
    def __init__(self):
        self.field = None
        self.string_field = ("X" * 64 + '\n') * 64

    def register_state(self, field, agent_body=np.ones((1,1))*C.HEAD, agent_pos=(7, 7)):
        ret_field = field.copy()
        rows, cols = agent_body.shape
        row, col = (agent_pos[0], agent_pos[1])
        ret_field[row:row+rows, col:col+cols] = agent_body
        self.field = ret_field
        self.string_field = field_to_string(self.field)

    def print_last_state(self):
        print(self.string_field)
