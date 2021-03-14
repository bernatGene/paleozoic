from pangea import Pangea
from utils import viewer
import constants as C

pan = Pangea()
pan.agent.init_model(pan.perception())
vie = viewer.Viewer()
vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()

print("Orientation: ", pan.agent_ori )

r, s = pan.perform_action(C.ROTATE)
pan.agent.react(r, s)
print(r, pan.agent.perception)

print("Orientation: ", pan.agent_ori )

for i in range(15):
    r, s = pan.perform_action(C.FORWARD)
    pan.agent.react(r, s)
    print(r)
    print(viewer.field_to_string(pan.agent.perception))

vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()


pan.run_day()
vie.register_state(pan.labyrinth.field, pan.agent.body, pan.agent_pos)
vie.print_last_state()

