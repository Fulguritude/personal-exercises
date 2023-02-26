import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
import ProportionBySport as pbs


df = fl.load("../athlete_events.csv")
print(str(pbs.proportion_by_sport(df, 2004, 'Tennis', 'F')))