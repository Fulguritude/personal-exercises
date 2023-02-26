import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
from SpatioTemporalData import SpatioTemporalData

df = fl.load("../athlete_events.csv")
st = SpatioTemporalData(df)

print(str(st.when('Atlanta')))
print(str(st.where(1936)))
print(str(st.where(1996)))