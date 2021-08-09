from src.utils import viewer

v = viewer.Viewer(None, None)
v.load_day("tests/test_pangea_4Agents_torch_predation1.day")
v.print_day(period=0.5)
