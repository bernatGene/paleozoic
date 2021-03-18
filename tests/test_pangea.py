from src.pangea import Pangea

pan = Pangea()

pan.run_day(report_steps=True)
pan.viewer.save_day(day_name="test_pangea1")
