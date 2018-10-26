import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV


dataset = pd.read_csv("F:/201810_ML_Learning/J1_annually_data_Handled_Reduction.csv")
print(dataset.columns)
a=6
b=8
print(dataset.columns[a])
print(dataset.columns[b])
X = dataset.iloc[:, [a,b]].values
y = dataset.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


grid = GridSearchCV(SVC(), param_grid={"C":[0.1], "gamma": [10, 1, 0.1]}, cv=4)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

print(grid.best_params_["C"])
print(grid.best_params_["gamma"])

classifier = SVC(kernel="rbf",gamma=grid.best_params_["gamma"], C=grid.best_params_["C"])
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classifier.score(X_test, y_test))

X_set, y_set = X_train, y_train
print(X_set.shape)
h=0.01
A=np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = h)
B=np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = h)
X1, X2 = np.meshgrid(A,B)
Z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
Z=Z.reshape(X1.shape)
plt.contourf(X1, X2, Z,alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

print(np.unique(y_set))
print(y_set.shape)
print(X_set.shape)
for i, j in enumerate(np.unique(y_set)):
    print(X_set[y_set == j, 0].shape)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],             #X_set[y_set == j, 0] 温度列，当
                c = ListedColormap(('red', 'green'))(i), label = j,s=1)
plt.title('SVM (Training set)')
plt.xlabel(r'${}$'.format(dataset.columns[a]))
plt.ylabel(r'${}$'.format(dataset.columns[b]))
plt.legend()
plt.show()