# pyro-ppl==0.3.0
# pytorch=1.0.0

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from torch.autograd import Variable
import torch.nn as nn
import pyro
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.infer import Trace_ELBO
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from importlib import reload


# Hyper Parameter
training_step = 1000

bnn = []
features = []


# 均方误差
def RMSE(Y_pred, Y_true):
    mse = 0
    for i in range(Y_pred.shape[0]):
        mse += np.sqrt((Y_pred.iloc[i, 0] - Y_true.iloc[i, 0]) * (Y_pred.iloc[i, 0] - Y_true.iloc[i, 0]) +
                       (Y_pred.iloc[i, 1] - Y_true.iloc[i, 1]) * (Y_pred.iloc[i, 1] - Y_true.iloc[i, 1]))
    mse = mse / Y_pred.shape[0]
    return mse


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
        self.PCA_N_COMPOENTS = False

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
        # self.scaler_for_OHLVCattP = MinMaxScaler(feature_range=(0, 1))
        # self.scaler_for_price = MinMaxScaler(feature_range=(0, 1))
        self.scaler_for_OHLVCattP = StandardScaler()
        self.scaler_for_price = StandardScaler()
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

        df.iloc[: -self.time_step, -1:] = self.scaler_for_price.fit_transform(
            np.float64(df.iloc[: -self.time_step, -1:]))
        df.iloc[-self.time_step:, -1:] = self.scaler_for_price.transform(
            np.float64(df.iloc[-self.time_step:, -1:]))
        self.df = df

    def generator(self):
        X = []
        Y = []
        feature_list = features
        for i in range(
                len(self.df) - self.train_time_step - self.time_step + 1):  # windows_size时保证有windows_size个数据是是连续的
            X.append(self.df.iloc[i: i + self.train_time_step, feature_list].values)
            Y.append(self.df.iloc[i + self.train_time_step: i + self.train_time_step + self.time_step, -1:].values)
        self.X_train = np.array(X[:-self.time_step])
        self.Y_train = np.array(Y[:-self.time_step])
        self.X_test = np.array(X[-self.time_step:])
        self.Y_test = np.array(Y[-self.time_step:])
        total_batch = int(self.X_train.shape[0])
        print("Number of batches per epoch:", total_batch)

    def __call__(self, testdate, pca_components=False):
        self.testdate = testdate
        self.predict_test_point = (datetime.strptime(self.testdate, '%Y-%m-%d') - datetime.strptime('2016-01-01',
                                                                                                    '%Y-%m-%d')).days  # 注意self.df的起始时间为2016-01-01
        self.from_point = self.predict_test_point - 13 * 30  # 计算周期的起点
        self.to_point = self.predict_test_point + 1 * self.time_step  # 计算周期的终点
        self.preprocessing()
        self.generator()
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


def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, 8*n_hidden)
        self.hidden3 = torch.nn.Linear(8 * n_hidden, 4 * n_hidden)
        self.hidden4 = torch.nn.Linear(4 * n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, 1)  # output layer
        self.out_14 = torch.nn.Linear(60, 14)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden2(x)

        x = self.hidden3(x)
        x = self.hidden4(x)

        x = self.out(x)
        x = self.out_14(x.view(x.shape[0], x.shape[1]))
        x = x.view(-1, 1)
        return x


class Batch_Generator():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        '''传入的结构为:[batch_size, time_step, features_size]'''
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_count = 0
        self.np_random_seed = 0
        self.iter_count = 1
        self.batch_size = 256
        self.BATCH_INDEX = 0
        self.indexs = list(range(len(self.X_train)))

    def get_train(self):
        self.np_random_seed += 1
        np.random.seed(self.np_random_seed)

        def get_batch_indices(N, batch_size):
            all_batches = np.arange(0, N, batch_size)
            if all_batches[-1] != N:
                all_batches = list(all_batches) + [N]
            return all_batches

        # 用于选取X_train的哪一部分作为第1， 2， 。。。个batch
        if self.iter_count:
            self.all_batch_index = get_batch_indices(len(self.X_train), self.batch_size)
            print('A epoch contain {} iterations'.format(len(self.all_batch_index) - 1))  # 因为是from..to结构，最后一位没用
            self.iter_count = 0
        # 一个batch训练完后需要对数据进行重新筛选
        # 重置条件为BATCH_INDEX == 倒数第二个batch
        if self.BATCH_INDEX == (len(self.all_batch_index) - 1):
            self.indexs = np.random.permutation(range(len(self.X_train)))
            #                 self.X_train = self.X_train[indexs]
            #                 self.Y_train = self.Y_train[indexs]
            self.BATCH_INDEX = 0

        # print('current batch:{} / {}'.format(self.BATCH_INDEX, len(self.all_batch_index) - 1))
        batch_begin = self.all_batch_index[self.BATCH_INDEX]
        batch_end = self.all_batch_index[self.BATCH_INDEX + 1]
        X_out = np.array(self.X_train[self.indexs[batch_begin: batch_end]])
        Y_out = np.array(self.Y_train[self.indexs[batch_begin: batch_end]])
        self.BATCH_INDEX += 1
        return X_out, Y_out

    def get_train_(self):
        X = self.X_train[-1:]
        Y = self.Y_train[-1:]
        return X, Y

    def get_test(self):
        X = self.X_test[-1]
        Y = self.Y_test[-1]
        return X, Y


class BNN():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        """类似数的初始化，"""
        self.X_train, self.Y_train = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(Y_train))

        # 测试数据这时候导
        self.X_test, self.Y_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(Y_test))

        # 特征数
        self.first_layer = self.X_train.data.shape[2]
        self.second_layer = 25
        self.N = len(X_train)

    def build_bnn(self):
        self.softplus = nn.Softplus()
        self.regression_model = Net(self.first_layer, self.second_layer)

    def model(self, X_data, Y_data):
        data = bnn.X_train

        mu = Variable(torch.zeros(self.second_layer, self.first_layer)).type_as(data)
        sigma = Variable(torch.ones(self.second_layer, self.first_layer)).type_as(data)
        bias_mu = Variable(torch.zeros(self.second_layer)).type_as(data)
        bias_sigma = Variable(torch.ones(self.second_layer)).type_as(data)
        w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)

        mu2 = Variable(torch.zeros(1, self.second_layer)).type_as(data)
        sigma2 = Variable(torch.ones(1, self.second_layer)).type_as(data)
        bias_mu2 = Variable(torch.zeros(1)).type_as(data)
        bias_sigma2 = Variable(torch.ones(1)).type_as(data)
        w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)

        priors = {'hidden.weight': w_prior,
                  'hidden.bias': b_prior,
                  'predict.weight': w_prior2,
                  'predict.bias': b_prior2}

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.regression_model, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        # with pyro.plate("map", self.N, subsample=X_data):
        x_data = Variable(torch.Tensor(X_data))

        y_data = Variable(torch.Tensor(Y_data))
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample("obs", Normal(prediction_mean, Variable(torch.ones(x_data.shape[0] * 14)).type_as(data)), obs=y_data.squeeze())

    def guide(self, X_data_guide, Y_data):
        data = self.X_train
        w_mu = Variable(torch.randn(self.second_layer, self.first_layer).type_as(data.data), requires_grad=True)
        w_log_sig = Variable(0.1 * torch.ones(self.second_layer, self.first_layer).type_as(data.data),
                             requires_grad=True)
        b_mu = Variable(torch.randn(self.second_layer).type_as(data.data), requires_grad=True)
        b_log_sig = Variable(0.1 * torch.ones(self.second_layer).type_as(data.data), requires_grad=True)

        # register learnable params in the param store
        mw_param = pyro.param("guide_mean_weight", w_mu)
        sw_param = self.softplus(pyro.param("guide_log_sigma_weight", w_log_sig))
        mb_param = pyro.param("guide_mean_bias", b_mu)
        sb_param = self.softplus(pyro.param("guide_log_sigma_bias", b_log_sig))

        # gaussian guide distributions for w and b
        w_dist = Normal(mw_param, sw_param)
        b_dist = Normal(mb_param, sb_param)

        w_mu2 = Variable(torch.randn(1, self.second_layer).type_as(data.data), requires_grad=True)
        w_log_sig2 = Variable(0.1 * torch.randn(1, self.second_layer).type_as(data.data), requires_grad=True)
        b_mu2 = Variable(torch.randn(1).type_as(data.data), requires_grad=True)
        b_log_sig2 = Variable(0.1 * torch.ones(1).type_as(data.data), requires_grad=True)

        # register learnable params in the param store
        mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
        sw_param2 = self.softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
        mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
        sb_param2 = self.softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))

        # gaussian guide distributions for w and b
        w_dist2 = Normal(mw_param2, sw_param2)
        b_dist2 = Normal(mb_param2, sb_param2)

        dists = {'hidden.weight': w_dist,
                 'hidden.bias': b_dist,
                 'predict.weight': w_dist2,
                 'predict.bias': b_dist2}

        # overloading the parameters in the module with random samples from the guide distributions
        lifted_module = pyro.random_module("module", self.regression_model, dists)
        # sample a regressor
        return lifted_module()

    def optimizer_model(self):
        optim = Adam({"lr": 3e-4})
        self.svi = SVI(self.model, self.guide, optim, Trace_ELBO())


def run():
    global bnn
    batch_generator = Batch_Generator(DF.X_train, DF.Y_train, DF.X_test, DF.Y_test)
    X_train_batch, Y_train_batch = batch_generator.get_train()
    X_test_batch, Y_test_batch = batch_generator.get_test()
    bnn = BNN(X_train_batch, Y_train_batch,
              X_test_batch, Y_test_batch)
    bnn.build_bnn()
    bnn.optimizer_model()

    # 训练阶段开始
    # 记录所有训练过程中的损失
    train_loss_all = []
    for step in range(1, training_step + 1):
        epoch_loss = 0.0
        X_train_batch, Y_train_batch = batch_generator.get_train()
        X_test_batch = Variable(torch.Tensor(X_test_batch))
        Y_train_batch = Variable(torch.Tensor(Y_train_batch)).view(-1, 1)
        epoch_loss += bnn.svi.step(X_train_batch, Y_train_batch)

        if step % 100 == 0:
            print("\r{}, avg loss {}".format(step, epoch_loss / float(bnn.N)), end='')

    # # train_loss = sess.run(bnn.loss_op, )
    # print("Step: {}/{}, train loss:{}".format(step, training_step, train_loss))
    # train_loss_all.append(train_loss)
    # # if step % display_step == 0 or step == 1:

    # 预测阶段
    # predict_coor = pd.DataFrame(index=datareader.Y_test.index,  columns=['x', 'y'])

    preds = []
    for i in range(100):
        X_test_batch, Y_test_batch = batch_generator.get_test()
        X_test_batch = X_test_batch.reshape((1, X_test_batch.shape[0], X_test_batch.shape[1]))
        X_test_batch = Variable(torch.Tensor(X_test_batch))
        sampled_reg_model = bnn.guide(X_test_batch, Y_test_batch)
        pred = sampled_reg_model(X_test_batch).data.numpy().flatten()
        preds.append(pred)

    preds = np.array(preds)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0) / 10
    y_test = bnn.Y_test.data.numpy()
    x = np.arange(len(y_test))

    original = Y_test_batch
    original = DF.scaler_for_price.inverse_transform(original)
    mean = DF.scaler_for_price.inverse_transform(mean.reshape((1, -1))).ravel()
    plt.figure()
    plt.plot(original)
    plt.plot(x, mean, linestyle='--')
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, color='orange')
    plt.savefig(save_dir + '/bnn-{}'.format(run_time))
    plt.close()
    bnn_mse = mean_squared_error(mean, original)
    error_ntime.append(bnn_mse)
    print('bnn_mse', bnn_mse)
    logging.info(
        "Record predict value and para:\npredict_test_point: {}\ntime_step: {}\ntest_predict: {}\ntest_y: {}\n\n"
        "".format(DF.predict_test_point, DF.time_step, mean.ravel(), original.ravel()))
    return bnn_mse


if __name__ == '__main__':
    testdates = ['2017-10-26', '2017-11-24', '2018-04-11'] + \
                ['2017-10-30', '2018-04-29', '2018-07-22'] + \
                ['2017-12-27', '2018-02-02', '2018-03-10'] + \
                ['2018-03-05', '2018-01-07', '2018-08-01'] + \
                ['2017-06-15', '2017-08-13', '2018-09-21']
    feature_list_all = [[28], [1, 2, 3, 4, 28], [5, 6, 7, 8, 28],
                        [9, 28], [5, 6, 7, 8, 9, 28],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 28],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28],
                        list(range(1, 29))]
    features_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28]]
    for features in features_list:
        for date in testdates:
            DF = GetData()
            DF(date)
            # 记录误差
            error_ntime = []
            for run_time in range(1, 1 + 10):
                num = np.array(features).sum()
                save_dir = './bnn-result{}/'.format(num) + date
                os.makedirs(save_dir, exist_ok=True)
                logging.basicConfig(filename=save_dir + '/logger.log', level=logging.INFO)
                run()
            error_ntime = np.array(error_ntime)
            logging.info('Min MSE: {}  ,   Average MSE:{}, Time: {}'.format(error_ntime.min(), error_ntime.mean(),
                                                                            error_ntime.argmin()))
            logging.shutdown()
            reload(logging)
