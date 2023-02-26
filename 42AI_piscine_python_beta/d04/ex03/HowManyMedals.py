#import pandas as pd
#https://stackoverflow.com/questions/22391433/count-the-frequency-that-a-value-occurs-in-a-dataframe-column
#https://stackoverflow.com/questions/18327624/find-elements-index-in-pandas-series

def how_many_medals(df, name):
	player_df = df[df['Name'] == name]
	year_list = player_df['Year'].unique()
	medal_per_year_dic = {}
	for year in year_list:
		medal_dic = {'G': 0, 'S': 0, 'B': 0}
		player_year_df = player_df[player_df['Year'] == year]['Medal']
		counts = player_year_df.value_counts() #return a pd.Series object
		for i in range(len(counts)):
			if counts.index[i] == 'Gold':
				medal_dic['G'] = counts[i]
			elif counts.index[i] == 'Silver':
				medal_dic['S'] = counts[i]
			elif counts.index[i] == 'Bronze':
				medal_dic['B'] = counts[i]
		medal_per_year_dic[year] = medal_dic
	return medal_per_year_dic