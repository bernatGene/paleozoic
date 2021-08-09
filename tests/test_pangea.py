from src.pangea import Pangea

png = Pangea((64, 32), food_limit=0)
png.report_step()
png.agent_intersections(0)


png.run_day(max_steps=20, report_steps=True, report_progress=True)