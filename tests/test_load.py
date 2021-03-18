from src.utils import viewer

v = viewer.Viewer(None, None)
v.load_day("test_pangea_4Agents")
v.print_day(period=0.1)