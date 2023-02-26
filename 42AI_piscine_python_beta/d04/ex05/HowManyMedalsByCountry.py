#https://en.wikipedia.org/wiki/France_at_the_2008_Summer_Olympics

def how_many_medals_by_country(df, country_name):
	country_df = df[df['Team'] == country_name]
	year_list = country_df['Year'].unique()
	medal_per_year_dic = {}
	for year in year_list:
		medal_dic = {'G': 0, 'S': 0, 'B': 0}
		country_year_df = country_df[country_df['Year'] == year]
#		if year == 2008:
#			print(str(country_year_df[country_year_df['Medal'] == 'Bronze']) + "\n")
		country_year_df.drop_duplicates(['Event', 'Medal', 'Sex'], inplace = True)
#		if year == 2008:
#			print(str(country_year_df[country_year_df['Medal'] == 'Bronze']))
		country_year_df = country_year_df['Medal']
		counts = country_year_df.value_counts() #return a pd.Series object
		for i in range(len(counts)):
			if counts.index[i] == 'Gold':
				medal_dic['G'] = counts[i]
			elif counts.index[i] == 'Silver':
				medal_dic['S'] = counts[i]
			elif counts.index[i] == 'Bronze':
				medal_dic['B'] = counts[i]
		medal_per_year_dic[year] = medal_dic
	return medal_per_year_dic