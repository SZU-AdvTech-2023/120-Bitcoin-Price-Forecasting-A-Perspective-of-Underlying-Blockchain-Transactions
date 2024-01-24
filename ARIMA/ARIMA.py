import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import logging
from importlib import reload
import os



class GetData():
    def __init__(self):
        self.time_step = 14
        self.train_time_step = 60
        self.df = 0
        self.scaler_for_price = 0
        self.df_store_all_price_data = 0
        self.testdate = '2018-01-01'
        self.predict_test_point = (datetime.strptime(self.testdate, '%Y-%m-%d') - datetime.strptime('2016-01-01',
                                                                                         '%Y-%m-%d')).days  # 注意self.df的起始时间为2016-01-01
        self.from_point = self.predict_test_point - 13 * 30  # 计算周期的起点
        self.to_point = self.predict_test_point + 1 * self.time_step  # 计算周期的终点
        self.PCA_N_COMPOENTS = 6

    def preprocessing(self):
        # Price, Open, High, Low, Vol., Change %
        # 430, 427.1, 433.1, 417.1, 65.21K, 0.66 %
        df6 = pd.read_csv(
            r'../data/btc-price-daily-2010-2019/btc-price-daily-2010-2019/Bitcoin Historical Data - 2016.csv')
        df7 = pd.read_csv(
            r'../data/btc-price-daily-2010-2019/btc-price-daily-2010-2019/Bitcoin Historical Data - 2017.csv')
        df8 = pd.read_csv(
            r'../data/btc-price-daily-2010-2019/btc-price-daily-2010-2019/Bitcoin Historical Data - 2018.csv')
        df = pd.concat([df6, df7, df8])
        df = df[['Date', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Price']]
        df = df.reset_index(drop=True)
        for index in df.index:  # 数据格式为17k,16m、17,000的格式，需要进行转换
            split_time = datetime.strptime(df['Date'][index], '%b %d, %Y')
            df['Date'][index] = '-'.join([str(split_time.year), str(split_time.month), str(split_time.day)])
            df['Price'][index] = float(str(df['Price'][index]).replace(',', ''))
            df['Open'][index] = float(str(df['Open'][index]).replace(',', ''))
            df['High'][index] = float(str(df['High'][index]).replace(',', ''))
            df['Low'][index] = float(str(df['Low'][index]).replace(',', ''))
            try:
                df['Vol.'][index] = float(df['Vol.'][index].replace('K', '')) * 1000
            except:
                df['Vol.'][index] = float(df['Vol.'][index].replace('M', '')) * 100000
            df['Change %'][index] = float(df['Change %'][index].replace('%', '')) * 0.01
        df['Date'] = pd.DatetimeIndex(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 加入google trend
        bitcoin_trend = pd.read_csv('../data/bitcoin_googletrend.csv')
        bitcoin_trend = bitcoin_trend.rename(columns={'bitcoin': 'google_trend'})
        df = pd.concat((df, bitcoin_trend['google_trend']), axis=1)

        # 大小交易所之间的数据datedate,avgfee_2small,avgfee_2major,to_small_avgval,to_major_avgval
        df_feature = pd.read_csv('../data/statistic_day_attribute111.csv', index_col=0,
                                 usecols=['date', 'avgfee_2small', 'avgfee_2major', 'to_small_avgval',
                                          'to_major_avgval'])
        assert df_feature['2016-01-01':'2018-12-31'].shape[0] == df.shape[0]  # 判断长度是否一致，防止出错
        df_feature = df_feature['2016-01-01':'2018-12-31'].reset_index(drop=True)

        # 将特征合并到df中
        df = pd.concat((df, df_feature), axis=1)[
            ['Date', 'Open', 'High', 'Low', 'Price', 'Vol.', 'avgfee_2small', 'avgfee_2major', 'to_small_avgval',
             'to_major_avgval', 'google_trend']]
        df.rename(columns={'Price': 'Close'}, inplace=True)

        # 因为交易所之间的特征是使用比特币进行度量的， 这种单位没法很好得衡量每笔交易得真实价值（各个时期比特币得价值不同）
        # 故乘以价格
        df_feature = df.iloc[:, 6:10].mul(df.loc[:, 'Close'], axis=0)
        df_feature.rename(columns={
            'avgfee_2small': 'avgfee_2small_price', 'avgfee_2major': 'avgfee_2major_price',
            'to_small_avgval': 'to_small_avgval_price', 'to_major_avgval': 'to_major_avgval_price'}, inplace=True)
        df = pd.concat([df, df_feature], axis=1)

        df.iloc[:, 1:] = df.iloc[:, 1:].astype('float64')  # 换位pandas的数据格式
        # 对整个数据进行截断，仅保留要训练和测试的部分，故存储整个价格数据
        self.df_store_all_price_data = df[['Date', 'Close']]
        self.df_store_all_price_data.rename(columns={'Close': 'Price'}, inplace=True)
        # 将所有特征改为change,并且加入到df
        df_feature = df.iloc[:, list(range(1, len(df.columns)))].copy().pct_change()
        df_feature.columns = [name + '_change' for name in df.columns[1:]]
        df_feature = df_feature[[name for name in df_feature.columns if name != 'Close_change'] + ['Close_change']]
        # df_feature.rename(columns={'Close_change': 'Price'}, inplace=True)
        df = pd.concat([df, df_feature], axis=1)
        df = df[[col for col in df.columns if col != 'Close'] + ['Close']]
        df.rename(columns={'Close': 'Price'}, inplace=True)

        df = df.iloc[self.from_point: self.to_point, :]

        # 存储用于将change值还原为price的所需的变量
        self.change2price_lasttime_trueprice = df.iat[-self.time_step - 1, list(df.columns).index('Price')].copy()
        self.df_store_test_price = df.iloc[-self.time_step:, list(df.columns).index('Price')].copy().to_numpy()

        # 对数据进行归一化等
        self.scaler_for_OHLVCattP = MinMaxScaler(feature_range=(0, 1))
        # self.scaler_for_price = MinMaxScaler(feature_range=(0, 1))
        # self.scaler_for_OHLVCattP = StandardScaler()
        # self.scaler_for_price = StandardScaler()
        # 对训练集进行标准化, 不在这一步对价格进行标准化
        # df_price = df.iloc[:, -1:].copy()

        df.iloc[: -self.time_step, 1:-1] = self.scaler_for_OHLVCattP.fit_transform(
            np.float64(df.iloc[: -self.time_step, 1:-1]))

        df.iloc[-self.time_step:, 1:-1] = self.scaler_for_OHLVCattP.transform(
            np.float64(df.iloc[-self.time_step:, 1:-1]))

        # def sigmoid(x):
        #     y = (1 / (1 + np.exp(-Parameter.sigmoid_alpha * x)) - 1 / 2) * 2
        #     return y
        # df.iloc[:, 15:] = sigmoid(df.iloc[:, 15:])

        # df.iloc[: -self.time_step, -1:] = self.scaler_for_price.fit_transform(
        #     np.float64(df.iloc[: -self.time_step, -1:]))
        # df.iloc[-self.time_step:, -1:] = self.scaler_for_price.transform(
        #     np.float64(df.iloc[-self.time_step:, -1:]))
        self.df = df

    def __call__(self, testdate, pca_components=False):
        self.testdate = testdate
        self.predict_test_point = (datetime.strptime(self.testdate, '%Y-%m-%d') - datetime.strptime('2016-01-01',
                                                                                         '%Y-%m-%d')).days  # 注意self.df的起始时间为2016-01-01
        self.from_point = self.predict_test_point - 13 * 30  # 计算周期的起点
        self.to_point = self.predict_test_point + 1 * self.time_step  # 计算周期的终点
        self.preprocessing()
        if pca_components:
            # 进行训练集的主成分分析
            self.PCA_N_COMPOENTS = pca_components
            print('PCA: COMPONENTS-{}'.format(self.PCA_N_COMPOENTS))
            pca_price = PCA(n_components=self.PCA_N_COMPOENTS)
            # pca_price = PCA(n_components='mle')
            pca_price.fit(self.df.iloc[: -self.time_step, 1:])
            self.df_pca_train = pd.DataFrame(pca_price.transform(self.df.iloc[: -self.time_step, 1:]), copy=True)
            # 将训练集标准化的计算应用到测试集

            # 进行测试集的主成分分析
            self.df_pca_test = pd.DataFrame(pca_price.transform(self.df.iloc[-self.time_step:, 1:]), copy=True)
            self.df_pca = pd.concat([self.df_pca_train, pd.DataFrame(np.zeros(self.df_pca_test.shape))], axis=0)
            self.df_pca = pd.concat([self.df['Date'].reset_index(drop=True), self.df_pca.reset_index(drop=True),
                                     self.df['Price'].reset_index(drop=True)], axis=1)
            self.df = self.df_pca


def run_arima():
    train = DF.df.iloc[:-DF.time_step, -1]
    valid = DF.df.iloc[-DF.time_step:, -1]
    model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train)
    forecast = model.predict(n_periods=len(valid))
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    plt.plot(train[-60:], label='Train')
    plt.plot(valid, label='Valid')
    plt.plot(forecast, label='Prediction')
    plt.savefig(save_dir + '/Arima-{}'.format(run_time))
    #plt.close()
    arima_mse = mean_squared_error(valid,forecast)
    error_ntime.append(arima_mse)
    logging.info(
        "Record predict value and para:\npredict_test_point: {}\ntime_step: {}\ntest_predict: {}\ntest_y: {}\n\n"
        "".format(DF.predict_test_point, DF.time_step, forecast.values.ravel(), valid.values.ravel()))
    return arima_mse

if __name__ == '__main__':
    testdates = ['2017-10-26', '2017-11-24', '2018-04-11'] + \
                ['2017-10-30', '2018-04-29', '2018-07-22'] + \
                ['2017-12-27', '2018-02-02', '2018-03-10'] + \
                ['2018-03-05', '2018-01-07', '2018-08-01'] + \
                ['2017-06-15', '2017-08-13', '2018-09-21']

    for date in testdates:
        DF = GetData()
        DF(date)
        # 记录误差
        error_ntime = []
        for run_time in range(1, 1+10):
            save_dir = './test-arima-result/' + date
            os.makedirs(save_dir, exist_ok=True)
            logging.basicConfig(filename=save_dir + '/logger.log', level=logging.INFO)
            run_arima()
        error_ntime = np.array(error_ntime)
        logging.info('Min MSE: {}  ,   Average MSE:{}, Time: {}'.format(error_ntime.min(), error_ntime.mean(),
                                                               error_ntime.argmin()))
        logging.shutdown()
        reload(logging)