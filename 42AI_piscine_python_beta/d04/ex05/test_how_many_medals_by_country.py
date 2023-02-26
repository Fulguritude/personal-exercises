import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
import HowManyMedalsByCountry as hmm


df = fl.load("../athlete_events.csv")
print(str(hmm.how_many_medals_by_country(df, 'France')))