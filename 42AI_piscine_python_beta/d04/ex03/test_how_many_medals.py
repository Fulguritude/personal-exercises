import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
import HowManyMedals as hmm


df = fl.load("../athlete_events.csv")
print(str(hmm.how_many_medals(df, 'Paavo Johannes Aaltonen')))