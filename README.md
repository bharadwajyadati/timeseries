1) find an account , and its corresponding clients , location and other info
2) event of corona using SIR region level if possible 
3) predict sale forecast on this?


usage :


f = TimeSeriesAnaysis("hpq")
f.plot()  # plots all the values from the past till today
f.weekly_seasonality = True  # include weekly seasonality and check
f.modelling()  # modelling the existing and trying to fit
f.forecast(90)  # forcasting for next 1 quater
f.daily_seasonality = True  # to include daily seasonlity and check
f.weekly_seasonality = False  # upto you again if u want both ?
f.forecast(90)