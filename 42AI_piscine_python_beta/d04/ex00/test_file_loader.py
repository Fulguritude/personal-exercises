from FileLoader import FileLoader as fl

df = fl.load("../athlete_events.csv")

n = 5
fl.display(df, n)
print("")
fl.display(df, -n)