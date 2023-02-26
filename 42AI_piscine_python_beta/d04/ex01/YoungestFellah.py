#in pandas, axis 0 is rows and axis 1 is columns, unlike numpy
#https://chrisalbon.com/python/data_wrangling/filter_dataframes/
#http://www.datasciencemadesimple.com/get-minimum-value-column-python-pandas/

def youngest_fellah(df, year):
	#short_df = df.filter(items=['Sex', 'Age', 'Year'])
	short_df = df[df['Year'] == year]
	short_df = short_df[['Sex', 'Age']]
	m_df = short_df[short_df['Sex'] == 'M']['Age']
	f_df = short_df[short_df['Sex'] == 'F']['Age']
	dic = {'m': m_df.min(), 'f': f_df.min()}
	return dic
