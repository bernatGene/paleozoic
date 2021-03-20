from src.pangea import Pangea

pan = Pangea()
    # for a in pan.agents:
    #     a.load_model()

for d in range(2000):
    print(f"Day {d}:")
    pan.run_day(food_limit=1000 - d // 5, report_progress=True)


print("Day to be saved:")
pan.run_day(report_steps=True, max_steps=500, food_limit=1000, report_progress=True)
pan.viewer.save_day(day_name="test_pangea_4Agents2")
# for a in pan.agents:
#     a.save_model()

