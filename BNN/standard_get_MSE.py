import os
import re
import pandas as pd
import numpy as np
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def get_MSE():

    columns = [[10]]
    #通过listdir_LSTM = os.listdir()获得
    feature_combi = [path for path in os.listdir() if os.path.isdir(r'./' + path)]
    testdates = ['2017-10-26', '2017-11-24', '2018-04-11'] + \
                ['2017-10-30', '2018-04-29', '2018-07-22'] + \
                ['2017-12-27', '2018-02-02', '2018-03-10'] + \
                ['2018-03-05', '2018-01-07', '2018-08-01'] + \
                ['2017-06-15', '2017-08-13', '2018-09-21']

    #设置号索引
    index_testdates = [['上升'] * 3 + ['上凸'] * 3 + ['下凹'] * 3 + ['下降'] * 3 + ['平缓'] * 3, testdates]
    #用来存放最小均方误差的表
    df_min = pd.DataFrame(np.zeros((len(testdates), len(columns))), index=index_testdates, columns=[str(i) for i in columns])

    #用来存放平均均方误差的表
    df_avg = pd.DataFrame(np.zeros((len(testdates), len(columns))), index=index_testdates, columns=[str(i) for i in columns])

    # 用来存放最小均方误差的出现的标号的表
    df_min_index = pd.DataFrame(np.zeros((len(testdates), len(columns))), index=index_testdates, columns=[str(i) for i in columns])


    for feature, feature_name in zip(feature_combi, [str(i) for i in columns]):
        for testdate, cureve_shape_date,  in zip(testdates, list(zip(index_testdates[0], index_testdates[1]))):
            with open('./{}/{}/logger.log'.format(feature, testdate)) as f:
                for line in f:
                    if 'INFO:root:Min MSE:' in line:        #根据判断是不是最后一行
                        featch_data= re.findall(r'[\d\.]+', line)     #获取最小均方误差、平均均方误差、最小均方误差的出现的标号
                        df_min.loc[cureve_shape_date, feature_name] = np.int64(float(featch_data[0]))
                        df_avg.loc[cureve_shape_date, feature_name] = np.int64(float(featch_data[1]))
                        df_min_index.loc[cureve_shape_date, feature_name] = np.int64(featch_data[2])

    for df, df_name in zip([df_min, df_avg, df_min_index], ['df_min', 'df_avg', 'df_min_index']):
        try:
            with open('./{}_from_result.xlsx'.format(df_name)) as f:
                print('./{}_from_result.xlsx    存在，请保存或删除'.format(df_name))
        except:
            df.to_excel('{}_from_result.xlsx'.format(df_name))
            print('./{}_from_result.xlsx    保存成功'.format(df_name))

    pass


if __name__ == '__main__':
    get_MSE()