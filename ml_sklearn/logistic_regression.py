import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def run():
    # Breast Cancer Wisconsin dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                     'breast-cancer-wisconsin/wdbc.data', header=None)
    # y为字符型标签
    # 使用LabelEncoder类将其转换为0开始的数值型
    X, y = df.values[:, 2:], df.values[:, 1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0)

    pipe_lr = Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)

    print('Test accuracy: %.3f' % pipe_lr.score(X_test, y_test))