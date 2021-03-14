import labyrinth as lab
from utils import viewer

l = lab.Labyrinth()
v = viewer.Viewer()
v.register_state(l.field)
v.print_last_state()
