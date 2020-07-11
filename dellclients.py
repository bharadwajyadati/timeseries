import os
import pandas as pd


class DataAnalysis:

    def __init__(self, path):
        self.path = path
        pass

    def analysis(self):
        df = pd.read_csv(self.path, header=0)
        print(df.count())
        print(df.head())
        #columns = list(df.columns)
        #print(df.groupby('Opportunity ID').count())



d = DataAnalysis("Venafi History.csv")
d.analysis()
de = DataAnalysis("Venafi.csv")
de.analysis()
