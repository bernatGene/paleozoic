from src.utils import viewer

v = viewer.Viewer(None, None)
v.load_day("test_pangea2")
v.print_day(period=0.1)