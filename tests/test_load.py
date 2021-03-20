from src.utils import viewer

v = viewer.Viewer(None, None)
v.load_day("test_pangea_4Agents2")
v.print_day(period=0.05)