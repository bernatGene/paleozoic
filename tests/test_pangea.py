from src.pangea import Pangea
from src.utils import viewer
import src.constants as C
import numpy as np
import time

pan = Pangea()
pan.agent.dna = "#"
pan.agent.init_model(pan.perception())
pan.agent.load_model()
vie = viewer.Viewer()
vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()

print("Orientation: ", pan.agent_ori )

r, s = pan.perform_action(C.ROTATE)
pan.agent.react(r, s)
print(r, pan.agent.perception)

print("Orientation: ", pan.agent_ori )

for i in range(5):
    r, s = pan.perform_action(C.FORWARD)
    pan.agent.react(r, s)
    print(r)
    print(viewer.field_to_string(pan.agent.perception))

vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()


pan.run_day()
vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()


for i in range(1):
    if i % 100 == 0:
        pan.agent_pos = np.array([7, 7])
        pan.labyrinth.build_walls()
        pan.labyrinth.food_count = 0
        pan.labyrinth.grow_food()
    print(i)
    pan.run_day()

vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()

pan.agent_pos = np.array([7, 7])
pan.labyrinth.build_walls()
pan.labyrinth.food_count = 0
pan.labyrinth.grow_food()
pan.run_day(report_steps=True, max_steps=600)
print("day runned")
pan.viewer.save_day()
pan.agent.save_model()
