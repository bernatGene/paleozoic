import numpy as np
import pickle
import src.constants as C
import time
import sys
import gzip


def field_to_string(field, colormap=None):
    if colormap is None:
        colormap = {}
    text = ""
    # Prints column numbers
    for n in range(2, -1, -1):
        text += "   "
        for c in range(field.shape[1]):
            text += str((c // (10 ** n)) % 10)
        text += '\n'
    for r, row in enumerate(field):
        text += f"{r:03d}"   # Prints row numbers
        for c, item in enumerate(row):
            if (r, c) in colormap:
                text += C.COLORS[colormap[(r, c)]] + C.ASCII_DICT[item] + C.OFF
            else:
                text += C.ASCII_DICT[item]
        text += '\n'
    return text


class SavedDay:
    def __init__(self, field, agents_bodies, day):
        self.field = field.copy()
        self.agents_bodies = agents_bodies
        self.day = day


class Viewer:
    def __init__(self, field=None, agents_bodies=None):
        if field is not None:
            self.field = field.copy()
            self.field[field == C.FOOD] = C.NONE
        if agents_bodies is not None:
            self.agents_bodies = agents_bodies
            self.string_field = ""
        self.day = []

    def field_at_step(self, step):
        field = self.field.copy()
        for r, c in self.day[step]["food_pos"]:
            field[r, c] = C.FOOD
        colormap = {}
        energies = ""
        for i, (agent_id, p) in enumerate(self.day[step]["agents_pos"].items()):
            body = self.agents_bodies[agent_id].copy()
            ori = self.day[step]["agents_orients"][agent_id]
            ene = self.day[step]["agents_energy"][agent_id]
            vr, vc = C.body_coordinates_vector(body, ori)
            r = p[0] + vr
            c = p[1] + vc
            body = np.rot90(body, -C.ORIENTATIONS.index(ori))
            body_mask = (body != C.EMPTY)
            body_list = np.argwhere(body_mask)
            for (p0, p1) in body_list:
                if ene < 0:
                    colormap[(p0 + r, p1 + c)] = -1
                else:
                    colormap[(p0 + r, p1 + c)] = i % (len(C.COLORS) - 1)
            crop = field[r:r + body.shape[0], c:c + body.shape[1]]
            crop[body_mask] = C.EMPTY
            crop += body
            energies += f'Agent {i}: {ene:5.3f} | '
        heading = f'Day step: {step:04d} \n'
        return heading + energies + '\n' + field_to_string(field, colormap)

    def append_step(self, field, agents_pos=None, agents_energy=None, agents_orients=None):
        if agents_orients is None:
            agents_orients = {}
        if agents_energy is None:
            agents_energy = {}
        if agents_pos is None:
            agents_pos = {}
        food_pos = np.argwhere(field == C.FOOD)  # Do we really need to pass the whole field just for this?
        self.day.append({"agents_pos": agents_pos,  # Positions must be tuples
                         "agents_orients": agents_orients,
                         "agents_energy": agents_energy,
                         "food_pos": food_pos})

    def print_last_step(self):
        print(self.field_at_step(-1))

    def print_day(self, period=0.1):
        for step_idx in range(len(self.day)):
            str_step = self.field_at_step(step_idx)
            n_lines = str_step.count('\n')
            print(str_step)
            for _ in range(n_lines + 7):
                sys.stdout.write(C.RET_LINE)
            time.sleep(period)

    def save_day(self, day_name="unnamedDay.day"):
        with gzip.open(day_name, 'wb') as f:
            saved_day = SavedDay(self.field, self.agents_bodies, self.day)
            pickle.dump(saved_day, file=f, protocol=4)

    def load_day(self, day_name="src/unnamedDay.day"):
        with gzip.open(day_name, 'rb') as f:
            loaded_day = pickle.load(f)
            self.field = loaded_day.field
            self.agents_bodies = loaded_day.agents_bodies
            self.day = loaded_day.day
