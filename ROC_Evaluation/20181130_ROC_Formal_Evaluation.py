import numpy as np
np.random.seed(10)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=8000)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#  On totally LogisticRegression
LR = LogisticRegression(solver='lbfgs', max_iter=1000)
pipeline = make_pipeline(LR)
pipeline.fit(X_train, y_train)
A=pipeline.predict_proba(X_test)
y_pred_LR = pipeline.predict_proba(X_test)[:, 1]
fpr_LR, tpr_LR, _ = roc_curve(y_test, y_pred_LR)

# Supervised based on random forests
RF = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict_proba(X_test)[:, 1]
fpr_RF, tpr_RF, _ = roc_curve(y_test, y_pred_RF)
print("fpr_rf_lm{}".format(fpr_RF))
print(fpr_RF.shape)
print("tpr_rf_lm{}".format(tpr_RF))
print(tpr_RF.shape)

# Supervised transformation based on SVM
SVC=SVC(C=10,probability=True)
SVC.fit(X_train, y_train)
y_pre_svc = SVC.predict_proba(X_test)[:, 1]
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pre_svc)

plt.figure(1,figsize=(20,10))
ax1=plt.subplot(121)
ax2=plt.subplot(122)
tick_spacing = 0.1
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# plt.xticks(np.arange(0, 10, 1.0))
plt.show()

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_LR, tpr_LR, label=r'$Logistic\ Regression$')
# plt.plot(fpr_RF, tpr_RF, label=r'$Random\ Forest$')
# plt.plot(fpr_svc, tpr_svc, label=r'$SVC$')
# plt.xticks(fontsize=10)
# plt.xlabel(r'$False\ positive\ rate\ (\%)$',fontsize=15)
# plt.ylabel(r'$True\ positive\ rate\ (\%)$',fontsize=15)
# plt.title(r'$ROC\ curve$',fontsize=15)
# plt.legend(loc='best',fontsize=15)
#
# plt.subplot(122)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_LR, tpr_LR, label=r'$Logistic\ Regression$')
# plt.plot(fpr_RF, tpr_RF, label=r'$Random\ Forest$')
# plt.plot(fpr_svc, tpr_svc, label=r'$SVC$')
# plt.xticks(fontsize=10)
# plt.xlabel(r'$False\ positive\ rate\ (\%)$',fontsize=15)
# plt.ylabel(r'$True\ positive\ rate\ (\%)$',fontsize=15)
# plt.title(r'$ROC\ curve\ (zoomed\ in\ at\ top\ left)$',fontsize=15)
# plt.legend(loc='best',fontsize=15)
# plt.grid(True)
# # ax2.set_major_locator(xmajor_locator)
# plt.show()
