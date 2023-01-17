'''Data Environment'''
import pandas as pd
import numpy as np

class DataEnvironment:
    '''Create Database to be processed'''

    def __init__(self, symbol, api_granularity, granularity, window, lags, validation_share,
                 test_share, norm_select, featureset, startdate, enddate, price):

        self.symbol = symbol
        self.api_granularity = api_granularity
        self.granularity = granularity
        self.window = window
        self.lags = lags
        self.validation_share = validation_share
        self.test_share = test_share
        self.norm_select = norm_select
        self.featureset = featureset
        self.startdate = startdate
        self.enddate = enddate
        self.price = price
        self.cols = []
        self.features = []

        self._get_data()
        self._add_features()
        self._add_lags()
        self._data_split()
        self._normalize()


    def _get_data(self):
        '''Load price data'''

        # Activate one of the following versions A/B (Google Colab vs. local execution)

        # VERSION A: execute from Google Colab
        self.fn = f'tpq_algo/oanda_{self.symbol}_{self.startdate}_{self.enddate}_'
        self.fn += f'{self.api_granularity}_{self.price}.csv'
        self.fn = self.fn.replace(' ', '_').replace('-', '_').replace(':', '_')
        self.raw = pd.read_csv(self.fn, index_col=0, parse_dates=True)

        # VERSION B: execute local
        # self.fn = f'oanda_{self.symbol}_{self.startdate}_{self.enddate}_'
        # self.fn += f'{self.api_granularity}_{self.price}.csv'
        # self.fn = self.fn.replace(' ', '_').replace('-', '_').replace(':', '_')
        # self.raw = pd.read_csv(self.fn, index_col=0, parse_dates=True)

        # Use only closing price, rename column, resample date to desired granularity
        self.data = pd.DataFrame(self.raw['c'])
        self.data.columns = [self.symbol]
        self.data = self.data.resample(self.granularity, label='right').last().ffill()
        self.data.dropna(inplace=True)

    def _add_features(self):
        '''Add ML features (defined in configfile) to table'''

        if 'r' in self.featureset:
            self.data['r'] = np.log(self.data[self.symbol] / self.data[self.symbol].shift())
            self.data.dropna(inplace=True)
        if 'sma' in self.featureset:
            self.data['sma'] = self.data[self.symbol].rolling(self.window).mean()
        if 'min' in self.featureset:
            self.data['min'] = self.data[self.symbol].rolling(self.window).min()
        if 'max' in self.featureset:
            self.data['max'] = self.data[self.symbol].rolling(self.window).max()
        if 'mom' in self.featureset:
            self.data['mom'] = self.data['r'].rolling(self.window).mean()
        if 'vol' in self.featureset:
            self.data['vol'] = self.data['r'].rolling(self.window).std()
        if 'atr' in self.featureset:
            self.data[f'ATR_{self.window}_{self.symbol}'] = np.maximum(
                np.maximum(self.data[self.symbol].rolling(self.window).max() -
                self.data[self.symbol].rolling(self.window).min(),
                abs(self.data[self.symbol].rolling(self.window).max() -
                self.data[self.symbol].shift(1))),
                abs(self.data[self.symbol].rolling(self.window).min() -
                self.data[self.symbol].shift(1)))
            self.data['atr%'] = (self.data[f'ATR_{self.window}_{self.symbol}'] /
                                 self.data[self.symbol])

        self.data.dropna(inplace=True)
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)     #For upward movement set 1
        self.data['d'] = self.data['d'].astype(int)

    def _add_lags(self):
        '''Add lags for historical data'''

        for column in self.data:
            self.features.append(column)

        for feature in self.features:
            for lag in range(1, self.lags + 1):
                col = f'{feature}_lag_{lag}'
                self.data[col] = self.data[feature].shift(lag)
                self.cols.append(col)

        self.data.dropna(inplace=True)

    def _data_split(self):
        '''Split data into Train-/Test-/Validation-Dataset'''

        self.train_share = 1 - self.test_share - self.validation_share

        self.train_split = int(len(self.data) * self.train_share)
        self.test_split = int(len(self.data) * self.test_share)
        self.validation_split = int(len(self.data) * self.validation_share)

        self.train = self.data.iloc[:self.train_split].copy()
        self.validation = self.data.iloc[self.train_split:self.train_split +
                                         self.validation_split].copy()
        self.test = self.data.iloc[self.train_split + self.validation_split:].copy()
        self.trainval = pd.concat([self.train, self.validation])

    def _normalize(self):
        '''Normalize input data'''

        if self.norm_select:
            self.mu, self.std = self.train.mean(), self.train.std()
            self.train_ = (self.train - self.mu) / self.std
            self.test_ = (self.test - self.mu) / self.std
            self.validation_ = (self.validation - self.mu) / self.std
        else:
            self.train_ = self.train.copy()
            self.test_ = self.test.copy()
            self.validation_ = self.validation.copy()

    def datainfo(self):
        '''Return details on dataframe'''

        self.data.info()

    def plot_data(self):
        '''Plot dataframe'''

        self.data[self.symbol].plot(figsize=(15,10))