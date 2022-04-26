import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import os


def main():
    """
    决策树算法实现
    """
    data = []
    labels = []
    with open("./data/tree.txt") as ifile:
        for line in ifile:
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
            tokens = line.strip().split(" ")
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])

    x = np.array(data)  # 所要划分的样本特征集
    labels = np.array(labels)
    y = np.zeros(labels.shape)  # 所要划分的样本结果
    print(y)
    print(labels == 'fat')
    y[labels == "fat"] = 1
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 使用分类器分类 使用信息熵作为划分标准
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    # model.fit(): 实际上就是训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)。
    clf.fit(x_train, y_train)
    # 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
    # 本例中，身高的权重为0.25，体重为0.75  

    print("特征的影响力", clf.feature_importances_)

    pre_result = clf.predict(x)
    # print(x_train)
    print(pre_result)
    # print(y_train)
    # print(np.mean(pre_result == y_train))
    print("predict", pre_result)
    preps = clf.predict_proba(x_train)
    print("predict_proba", preps)

    # 准确率和召回率
    precision, recall, thresholds = precision_recall_curve(
        y_train, clf.predict(x_train))

    # 例子：
    # 某池塘有1400条鲤鱼，300只虾，300只鳖。现在以捕鲤鱼为目的。撒一大网，逮着了700条鲤鱼，200只虾，100只鳖。那么，这些指标分别如下：
    # 正确率 = 700 / (700 + 200 + 100) = 70%
    # 召回率 = 700 / 1400 = 50%
    # F值 = 70% * 50% * 2 / (70% + 50%) = 58.3%
    # 不妨看看如果把池子里的所有的鲤鱼、虾和鳖都一网打尽，这些指标又有何变化：
    # 正确率 = 1400 / (1400 + 300 + 300) = 70%
    # 召回率 = 1400 / 1400 = 100%
    # F值 = 70% * 100% * 2 / (70% + 100%) = 82.35%        
    # 正确率是评估捕获的成果中目标成果所占得比例；召回率，顾名思义，就是从关注领域中，召回目标类别的比例；而F值，则是综合这二者指标的评估指标，用于综合反映整体的指标。
    answer = clf.predict_proba(x)[:, 1]
    # y_pred=clf.predict(x_test)
    # print(answer)
    score = clf.score(x_test, y_test)
    # print("平均值", np.mean(answer == y_pred))
    print("平均值", score)

    print(classification_report(y, answer, target_names=['thin', 'fat']))

    model_path = "./models/clf_dsccision.model"
    # 保存模型
    joblib.dump(clf, model_path, compress=0)
    # ＃加载模型
    RF = joblib.load(model_path)

    # 应用模型进行预测  测试集测试
    result = RF.predict(x_test)
    print(x_test)
    print(result)
    # predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。所以每一行的和应该等于1
    print(RF.predict_proba(x_test))



if __name__ == '__main__':
    main()