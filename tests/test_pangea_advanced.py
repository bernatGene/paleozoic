from src.pangea import Pangea

pan = Pangea()
for a in pan.agents:
    print(a.dna, a.dna_cost)

for d in range(1500):
    print(f"Day {d}:")
    pan.run_day(food_limit=2500 - d // 5, report_progress=True)

print("Day to be saved:")
pan.run_day(report_steps=True, max_steps=500, food_limit=2500, report_progress=True)
pan.viewer.field = pan.labyrinth.field
pan.viewer.save_day(day_name="test_pangea_4Agents_torch_predation1")


