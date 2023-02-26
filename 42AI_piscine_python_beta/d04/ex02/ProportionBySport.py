#https://stackoverflow.com/questions/38390242/sampling-one-record-per-unique-value-pandas-python

import pandas as pd

def proportion_by_sport(df, year, sport, gender):
	part_df = df[df['Year'] == year]
	part_df = part_df[part_df['Sex'] == gender]
	part_sport_df = part_df[part_df['Sport'] == sport]
	part_df.drop_duplicates(['Name'], inplace = True)
	part_sport_df.drop_duplicates(['Name'], inplace = True)
	return part_sport_df.shape[0] / part_df.shape[0]