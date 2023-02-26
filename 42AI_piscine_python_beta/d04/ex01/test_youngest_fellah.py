import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
import YoungestFellah as yf

df = fl.load("../athlete_events.csv")

print(str(yf.youngest_fellah(df, 2004)))