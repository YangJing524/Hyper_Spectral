import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='Times New Roman',size=7)
#mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

variable = ['Univariate', 'LWC-VIs']  # 单变量或多变量，对应excel文件名
task = variable[1]  # 0指定单变量或1指定多变量
file_path = task + '.xlsx'
index = [0, 1, 2, 3, 4, 5]  # 表示第几份sheet
k = 10  # k折交叉验证，可自选
sheet_name = ['1st', '2nd', '3rd', '4th', '5th', 'All']  # 对应6份sheet
save_file_name = ['a', 'b', 'c', 'd', 'e', 'f']  # 所保存图片的前缀名称，分别对应1-5个sheet
data_list = []

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
for idx in index:

    data_df = pd.read_excel(file_path, sheet_name=sheet_name[idx])  # 选择对应的sheet加载数据
    #if task == 'Univariate':
        #labels = data_df['LWC']  # 获取标签值
        #features = data_df.drop(['LWC'], axis=1)  # 提取训练数据
        #if not os.path.exists('./单变量散点图预测'):
           # os.makedirs('./单变量散点图预测')
       # save_figure_dir = './单变量散点图预测/'
    if task == 'LWC-VIs':
        data_df.drop(data_df.columns[0], axis=1, inplace=True)  # 去除第一行无关项
        labels = data_df['LWC']  # 获取标签值
        features = data_df.drop(['LWC'], axis=1)  # 提取训练数据
        if not os.path.exists('./多变量散点图预测'):
            os.makedirs('./多变量散点图预测')
        save_figure_dir = './多变量散点图预测/'

    # 随机划分数据集，2/3作为建模集，1/3作为测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1/3, random_state=1)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    print("The dataset has {} data points with {} variables each.".format(*data_df.shape))
    print('-' * 100)


def plot_scatter(y_preds, y_tests, save_name):
    """绘制散点图并保存模型"""
    plt.figure()
    for i in range(len(y_preds)):
        plt.subplot(2, 3, i+1)
        plt.subplots_adjust(top=0.96, bottom=0.09, left=0.08, right=0.985, wspace=0.35, hspace=0.35)

        r2 = "$R^2 = ${}".format(round(r2_score(y_tests[i], y_preds[i]), 2))
        rmse = "RMSE = {}".format(round(np.sqrt(mean_squared_error(y_tests[i], y_preds[i])), 2))
        metrics = r2 + '\n' + rmse
        title = "({})".format(save_file_name[i], family= 'Times New Roman')

        # 绘制散点图
        p = plt.scatter(y_tests[i], y_preds[i], c='black', marker='.',s=10)

        # 设定x轴/y轴范围
        plt.axis([25, 65, 25, 65])

        # 将上、右边框去掉
        ax = plt.gca()
        #ax.spines['right'].set_color('white')
        #ax.spines['top'].set_color('white')



        # 绘制1:1的斜线
        plt.plot([25, 65], [25, 65], linestyle="--", color='grey', lw=0.5)
        plt.text(25, 55, metrics,)

        # 设定x/y轴标签
        plt.xlabel('Measured LCC' + '\n'+'%s' % title)
        plt.ylabel('Predicted LCC')
        #plt.tick_params(labelsize=50)
        #plt.xticks([])  # ignore xticks，不显示x轴标签

    plt.savefig(save_figure_dir + save_name + '.png', dpi=300)  # 保存到当前目录下, 可指定dpi=?和图片格式
    # plt.show()


def gird_search_model(classifies, params, names, scoring, X_train, X_test, y_train, y_test):
    """使用k折交叉验证以r2_score为指标进行网格搜索寻求最佳的模型参数"""

    model = GridSearchCV(classifies, params, cv=k, scoring=scoring, refit='r2', n_jobs=-1, return_train_score=True)
    fit = model.fit(X_train, y_train)  # 拟合建模集数据
    print('Model Name: %r' % names)
    print('Best cv_test_r2_score: %f using %s' % (fit.best_score_, fit.best_params_))
    y_pred_train = fit.best_estimator_.predict(X_train)  # 用训练器集合中最好的estimator预测y_train_pred
    y_pred_test = fit.best_estimator_.predict(X_test)  # 用训练器集合中最好的estimator预测y_test_pred

    train_score_lists = []
    test_score_lists = []
    score_lists = []
    model_metrics_name = [r2_score, mean_squared_error]  # 模型评价指标，与scoring相对应
    for matrix in model_metrics_name:  # 计算各个模型评价指标
        train_score = matrix(y_train, y_pred_train)  # 计算训练集的评价指标
        test_score = matrix(y_test, y_pred_test)  # 计算测试集的评价指标
        if matrix == mean_squared_error:
            train_score_lists.append(np.sqrt(train_score))  # 把训练集的各个模型指标放在同一行
            test_score_lists.append(np.sqrt(test_score))  # 把测试集的各个模型指标放在同一行
        else:
            train_score_lists.append(train_score)  # 把训练集的各个模型指标放在同一行
            test_score_lists.append(test_score)  # 把测试集的各个模型指标放在同一行
    score_lists.append(train_score_lists)  # 合并训练集和测试集的结果（便于展示）
    score_lists.append(test_score_lists)  # 合并训练集和测试集的结果（便于展示）
    score_df = pd.DataFrame(score_lists, index=['train', 'test'], columns=['r2', 'rmse'])  # 将结果显示为列表格式
    print('EVALUATE_METRICS:')
    print(score_df)
    print('-'*100)
    return score_lists, y_pred_test


def main():
    # 针对不同的数据，应该进行相应的参数调整，下面仅给个示例
    names = ['Support Vector Machine ', 'Random Forest']
    classifiers = [SVR(), RandomForestRegressor()]
    parameter_svr = {'kernel': ['rbf'],             #'poly', 'linear'
                     'C':[1000],                #[0.0001,0.001,0.01,1,10,100,1000,10000],
                     'gamma':[10]}               #[0.0001,0.001,0.01,1,10,100,1000,10000]}  # 惩罚系数,1000, 5000, 10000 , 15000, 20000, 25000, 50000, 75000, 100000
    parameter_rfr = {'n_estimators':[500],'max_features':[1]}       #range(1, 10, 1)
    parameters = [parameter_svr, parameter_rfr]
    scoring = {'r2': 'r2', "rmse": "neg_mean_squared_error"}
    print(parameter_svr,parameter_rfr)
    train_score_list = []
    test_score_list = []
    y_test_pred_list = []  # 用列表存放每个模型对测试集的测试结果

    for idx in index:
        X_train = X_train_list[idx]
        X_test = X_test_list[idx]
        y_train = y_train_list[idx]
        y_test = y_test_list[idx]
        print("第{}个sheet：".format(idx+1))
        for clf, param, name in zip(classifiers, parameters, names):
            score_list, y_test_pred = gird_search_model(clf, param, name, scoring, X_train, X_test, y_train, y_test)
            train_score_list.append(score_list[0])
            test_score_list.append(score_list[1])
            y_test_pred_list.append(y_test_pred)
    print("y_test_pred_list_shape:", np.shape(y_test_pred_list))

    svm_y_test_pred_list = []
    rf_y_test_pred_list = []
    for i in range(len(index)):
        print("第{}个sheet：".format(i + 1))
        svm_y_test_pred_list.append(y_test_pred_list[2*i])
        rf_y_test_pred_list.append(y_test_pred_list[2*i+1])

        _train_score_list = train_score_list[2*i:2*i+2]
        _test_score_list = test_score_list[2*i:2*i+2]
        train_score_df = pd.DataFrame(_train_score_list, index=names, columns=['r2_score', 'rmse_score'])
        test_score_df = pd.DataFrame(_test_score_list, index=names, columns=['r2_score', 'rmse_score'])
        print('TRAIN_SCORE:'), print(train_score_df, '\n'), print('TEST_SCORE:'), print(test_score_df)
        # 保存excel结果
        train_score_df.to_excel(save_figure_dir + save_file_name[i] + '.xls', sheet_name=save_file_name[i])


    plot_scatter(svm_y_test_pred_list, y_test_list, 'scatter' + '_svm_png')  # 保存支持向量机的散点图
    plot_scatter(rf_y_test_pred_list, y_test_list, 'scatter' + '_rf_png')  # 保存随机森林的散点图


if __name__ == "__main__":
    main()
