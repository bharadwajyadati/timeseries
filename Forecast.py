import yfinance as yf
import pandas as pd
import numpy as np
import fbprophet
import pytrends
import matplotlib
import matplotlib.pyplot as plt
from pytrends.request import TrendReq


__version__ = 0.1


class TimeSeriesAnaysis:

    """

     TIme series forecasting model basing on facebook Prophet api (dumb hack i know!!)
     reasons for choosing
        1) super fast (after seeing ARIMA models :P )
        2) "Addtive model" rather than "Generative model" (like ARIMA which requires p,q,d to fit after seeing acf and pacf which are very high)
        3) includes seasonality and outliers
        4) dont you think its similar to aviso model ,we can keep forecasting everyday ??

     TODO: need to even tinker with "stan" too

    """
    """
        constructor with tinker value from yahoo finance and sets all the required values
        for the model and to forecast
        :params ticker: ticker (not case sentitive)
        :type: string


        TODO: give the company name , have a lookup and check the ticker name and use it
    """

    def __init__(self, ticker):

        ticker = ticker.upper()
        self.symbol = ticker

        try:
            company = yf.Ticker(ticker)
            # place holder for sector wise analysis
            self.sector = company.get_info()['sector']
            company = company.history(period="max")
        except Exception as e:
            print(
                'Error Retrieving Data from yahoo finance , probably you cant acess it ')
            print(e)
            return

        # create date as index
        df = company.reset_index(level=0)

        # Columns required for prophet
        df['ds'] = df['Date']

        if ('Adj. Close' not in df.columns):
            df['Adj. Close'] = df['Close']
            df['Adj. Open'] = df['Open']

        df['y'] = df['Adj. Close']
        df['Daily Change'] = df['Adj. Close'] - df['Adj. Open']

        # make a copy of dataframe
        self.df = df.copy()

        # for quick query of date and for plotting
        self.min_date = min(df['Date'])
        self.max_date = max(df['Date'])

        # recent close price for the company
        self.recent_price = float(
            self.df.loc[self.df.index[-1], 'y'])

        # find the max priee in entire history
        self.max_price = np.max(self.df['y'])
        self.min_price = np.min(self.df['y'])

        # and the corresponding dates of max and min values
        self.min_price_date = self.df[self.df['y']
                                      == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.df[self.df['y']
                                      == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]

        # training data history , can be tweaked , presently set to 10
        self.training_data_years = 10

        # these variables are for fbprophet setting them to false by default ,enable them if required explictly for seasonality
        self.changepoint_prior_scale = 0.05
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

        print('{} stock data from  {} to {}.'.format(self.symbol,
                                                     self.min_date,
                                                     self.max_date))

    """
        reseting the graph in case if some changes are overridden
        * static method
    """
    @staticmethod
    def reset_plot():

        # Restore default parameters
        matplotlib.rcdefaults()

        # default fig size for all plots
        matplotlib.rcParams['figure.figsize'] = (15, 10)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'

    """
         Method to linearly interpolate prices for modeling

        :param dataframe: df for  resampling
        :type: DataFrame

    """

    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')

        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe

    """
        Explictly remove weekends to make accurate predictions

        :param dataframe: df to remove weekend data
        :return: DataFrame

        :returns: dataframe after removing the weekend values
        :rtype: DataFrame

    """

    def remove_weekends(self, dataframe):

        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)

        weekends = []

        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)

        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)

        return dataframe

    """
        General purpose plotting fuction for a dataframe using matplotlib
    """

    def plot(self):
        # reset bfore plotting and then plot so figsize is prefect for notebook
        self.reset_plot()

        df = self.df.copy()
        plt.plot(df['Date'], df['Close'])
        plt.xlabel('Date')
        plt.ylabel('US $')
        plt.title('{} stock data from  {} to {}'.format(self.symbol,
                                                        self.min_date.year,
                                                        self.max_date.year))
        plt.show()

    """
        creating a time series forecasting model using prophect for additive models with seasonality and outliers

        :param days: if need to predict future in while modelling use this but try other api
        :type: int
        :param resample: if linear interploation is needed use this
        :type: boolean

    """

    def modelling(self, days=0, resample=False):

        # reset the plot since we need to plot after creating model
        self.reset_plot()

        # create a model using fb library with default params but we may add seasonality

        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,
                                  weekly_seasonality=self.weekly_seasonality,
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)

        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Fit on the data for training years

        train_data = self.df[self.df['Date'] > (
            self.max_date - pd.DateOffset(years=self.training_data_years))]

        if resample:
            train_data = self.resample(train_data)

        model.fit(train_data)

      # Make and predict for next year with predict dataframe
        forecast = model.make_future_dataframe(periods=days, freq='D')
        forecast_df = model.predict(forecast)

        if days > 0:
            # Print the predicted price
            print('Predicted Price on {} = ${:.2f}'.format(
                forecast_df.loc[forecast_df.index[-1], 'ds'], forecast_df.loc[forecast_df.index[-1], 'yhat']))

            title = '%s Past & Future of Stock Price' % self.symbol
        else:
            title = '%s Prediction of existing Stock Price ' % self.symbol

        # modelling is done now plotting time ! :D

        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(train_data['ds'], train_data['y'], 'ko-',
                linewidth=1.4, alpha=0.8, ms=1.8, label='Existing')

        # Plot the predicted values
        ax.plot(forecast_df['ds'], forecast_df['yhat'], 'forestgreen',
                linewidth=2.4, label='ForeCasting')

        # Plot the uncertainty interval as ribbon
        ax.fill_between(forecast_df['ds'].dt.to_pydatetime(), forecast_df['yhat_upper'], forecast_df['yhat_lower'], alpha=0.3,
                        facecolor='g', edgecolor='k', linewidth=1.4, label='Confidence Interval of Models')

        # Plot formatting
        plt.legend(loc=2, prop={'size': 10})
        plt.xlabel('Date')
        plt.ylabel('Stock Price in US $')
        plt.grid(linewidth=0.6, alpha=0.6)
        plt.title(title)
        plt.show()

        return model, forecast_df

    """
        Equivilant to statsmodel time series decomposition

        Function to decompose time series into trend, seasonality and plot them !

    """

    def seasonality_decomposition(self):

        self.reset_plot()
        if self.weekly_seasonality:
            plt.title("with  seasonality")
        else:
            plt.title("without seasonality")

        model, modelled_data = self.modelling()
        model.plot_components(modelled_data)
        plt.show()

    """
        Forceasting fuction to predict how many days we need to forecast basing on trainng from past years

        :param days: no of days to forecast by default its a month can go upto future quaters as well
        :type: int

        :returns: predicted future forecast value dataframe
        :type: DataFrame

    """

    def forecast(self, days=30):

        # train data from the past few years defefind by training_data_years
        train = self.df[self.df['Date'] > (
            max(self.df['Date']) - pd.DateOffset(years=self.training_data_years))]

        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,
                                  weekly_seasonality=self.weekly_seasonality,
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)

        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        model.fit(train)

        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future_df = model.predict(future)

        # Only concerned with future dates since we are doing a forecasting
        future_df = future_df[future_df['ds'] >= max(self.df['Date'])]

        # Remove the weekends
        future_df = self.remove_weekends(future_df)

        # Calculate whether increase or not
        future_df['diff'] = future_df['yhat'].diff()

        future_df = future_df.dropna()

        # Find the prediction direction and create separate dataframes
        future_df['direction'] = (future_df['diff'] > 0) * 1

        # Rename the columns for presentation
        future_df = future_df.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change',
                                              'yhat_upper': 'upper', 'yhat_lower': 'lower'})

        future_increase = future_df[future_df['direction'] == 1]
        future_decrease = future_df[future_df['direction'] == 0]

        # Print out the dates
        print('\nPredicted Increase: \n')
        print(
            future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])

        print('\nPredicted Decrease: \n')
        print(
            future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])

        self.reset_plot()

        # Set up plot
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12

        # Plot the predictions and indicate if increase or decrease
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the estimates
        ax.plot(future_increase['Date'], future_increase['estimate'],
                'g^', ms=12, label='Pred. Increase')
        ax.plot(future_decrease['Date'], future_decrease['estimate'],
                'rv', ms=12, label='Pred. Decrease')

        # Plot errorbars
        ax.errorbar(future_df['Date'].dt.to_pydatetime(), future_df['estimate'],
                    yerr=future_df['upper'] - future_df['lower'],
                    capthick=1.4, color='k', linewidth=2,
                    ecolor='darkblue', capsize=4, elinewidth=1, label='Pred with Range')

        # Plot formatting
        plt.legend(loc=2, prop={'size': 10})
        plt.xticks(rotation='45')
        plt.ylabel('Predicted Stock Price (US $)')
        plt.xlabel('Date')
        plt.title('Predictions for %s' % self.symbol)
        plt.show()

        return future_df

