import src.labyrinth as lab
import src.trilobit as tri
import src.constants as C
from src.utils import viewer

l = lab.Labyrinth()
print("Test 1: Test creature")
t2 = tri.Trilobit(dna="#+0----+---o")
print(t2.dna)
t2.build_body()
body = t2.body
agents_bodies = {"a": body}

v = viewer.Viewer(l.field, agents_bodies)
a_pos = {"a": (7, 7)}
a_ene = {"a": 0}
a_ori = {"a": C.NORTH}
v.append_step(l.field, {"a": [7, 7]}, {"a": 0}, {"a": C.NORTH})
v.print_last_step()
print("Rotations")

last_good = -1
for i in range(150):
    print("new iter")
    a_pos["a"] = ((a_pos["a"][0] + i % 2), (a_pos["a"][1] + (i + 1) % 2))
    a_ori["a"] = C.ORIENTATIONS[i % 4]
    print(f"iter {i}, {last_good}, {a_pos['a']}")
    val, rew = l.valid_position(agents_bodies["a"], a_pos["a"], a_ori["a"])
    if val:
        print("checked passed")
        last_good = i
        print(agents_bodies["a"], a_pos["a"], a_ori["a"])
        v.append_step(l.field, a_pos, a_ene, a_ori)
    # v.print_last_step()

v.save_day()
