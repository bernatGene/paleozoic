from src.pangea import Pangea

pan = Pangea()

# for a in pan.agents:
#     a.load_model()

for d in range(5000):
    pan.run_day(food_limit=1500 - d // 5)
    for i, a in enumerate(pan.agents):
        print(f'Agent {i}, overall reward of {a.overall_reward:.2f}')

pan.run_day(report_steps=True, max_steps=500)
pan.viewer.save_day(day_name="test_pangea2")
for a in pan.agents:
    a.save_model()

