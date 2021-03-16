import src.trilobit as tri
from src.utils import viewer
from src.assets import tiles
from src import constants as C
import numpy as np

field = tiles.TILE0
field = np.concatenate((field, field, field, field), axis=1)
v = viewer.Viewer(field, None)

print("Test 1: Test creature")
t2 = tri.Trilobit(dna="#+0----+---o")
print(t2.dna)
t2.build_body()
body = t2.body
v.agents_bodies = {"a": body, "b": body, "c": body, "d": body}
print(body)
print(viewer.field_to_string(body))
v.append_step(field, {"a": (7, 7)}, {"a": 0}, {"a": C.NORTH})
v.print_last_step()
print("Rotations")
v.append_step(field, {"a": (7, 7), "b": (7, 7+16), "c": (7, 7+31), "d": (7, 7+47)},
              {"a": 0, "b": 0, "c": 0, "d": 0},
              {"a": C.NORTH, "b": C.EAST, "c": C.SOUTH, "d": C.WEST})
v.print_last_step()

for i in range(50):
    v.append_step(field, {"a": (7, 7), "b": (7, 7 + 16), "c": (7, 7 + 31), "d": (7, 7 + 47)},
                  {"a": 0, "b": 0, "c": 0, "d": 0},
                  {"a": C.ORIENTATIONS[i % 4],
                   "b": C.ORIENTATIONS[(i + 1) % 4],
                   "c": C.ORIENTATIONS[(i + 2) % 4],
                   "d": C.ORIENTATIONS[(i + 3) % 4]})

v.save_day()


# t2 = tri.Trilobit(dna="#o++++++++0o++0++o-++--+-+-++-++-+-+---0-o++++--o0+o0+00+--+-+-+-+---0")
# print(t2.dna)
# t2.build_body()
# text = viewer.field_to_string(t2.body)
# print(text)
