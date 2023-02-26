class SpatioTemporalData():

	def __init__(self, df):
		self.df = df

	def when(self, location):
		return self.df[self.df['City'] == location]['Year'].unique()

	def where(self, year):
		return self.df[self.df['Year'] == year]['City'].unique()