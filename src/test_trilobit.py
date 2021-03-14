import trilobit as tri
from utils import viewer

t = tri.Trilobit()
v = viewer.Viewer()

t.init_model(None)
t.build_body()

t2 = tri.Trilobit(dna="#o++-o-----+----+---0")
print(t2.dna)
t2.build_body()
text = viewer.field_to_string(t2.body)
print(text)

t2 = tri.Trilobit(dna="#o++++++++0o++0++o-++--+-+-++-++-+-+---0-o++++--o0+o0+00+--+-+-+-+---0")
print(t2.dna)
t2.build_body()
text = viewer.field_to_string(t2.body)
print(text)