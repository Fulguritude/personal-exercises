import sys
sys.path.insert(1, '../ex00/')

from FileLoader import FileLoader as fl
from MyPlotLib import MyPlotLib as mpl


df = fl.load("../athlete_events.csv")
#mpl.histogram(df, ['Weight', 'Height'])
mpl.density(df, ['Weight', 'Height'])
#mpl.pair_plot(df, ['Weight', 'Height'])
#mpl.box_plot(df, ['Weight', 'Height'])