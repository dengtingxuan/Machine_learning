import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("F:/201810_ML_Learning/J1_annually_data_Handled_Reduction.csv")
print(dataset.columns)
# a=3
# b=4
# print(dataset.columns[a])
# print(dataset.columns[b])
X = dataset.iloc[:, 0:17].values
y = dataset.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)
print(X_train)
print(X_test)


classifier = LogisticRegression(verbose=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classifier.score(X_test, y_test))
print(dataset.columns)
print(classifier.coef_)
dict={"key":dataset.columns, "values":classifier.coef_}
print(dict)

# X_set, y_set = X_train, y_train
# print(X_set.shape)
# h=0.001
# A=np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max(), step = h)
# B=np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max(), step = h)
# X1, X2 = np.meshgrid(A,B)
# Z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
# Z=Z.reshape(X1.shape)
# plt.contourf(X1, X2, Z,alpha = 0.4, cmap = ListedColormap(('red', 'green')))
#
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
#
# print(np.unique(y_set))
# print(y_set.shape)
# print(X_set.shape)
# for i, j in enumerate(np.unique(y_set)):
#     print(X_set[y_set == j, 0].shape)
#
# marker=["x","+"]
#
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],             #X_set[y_set == j, 0] 温度列，当
#                 c = ListedColormap(('red', 'green'))(i),marker=marker[j], label = j, s=30)
# plt.title('SVM (Training set)')
# plt.xlabel('{}'.format(dataset.columns[a]))
# plt.ylabel('{}'.format(dataset.columns[b]))
# plt.legend()
# # plt.savefig("")
# plt.show()
