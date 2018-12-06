import numpy as np
# np.random.seed(10)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline

n_estimator = 10
dataset = pd.read_csv("F:/201810_ML_Learning\All_Data/Sum_OF_All_Data_KMJ1_7.csv")
X = dataset.iloc[:, 0:17].values
y = dataset.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=True)
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=True)
"""Reduce Training Set"""

"""On totally LogisticRegression"""
Time_LR_Start = time.time()
LR = LogisticRegression(solver='lbfgs', max_iter=1000)
pipeline = make_pipeline(LR)
pipeline.fit(X_train, y_train)
Time_LR_End = time.time()
A = pipeline.predict_proba(X_test)
y_pred_LR = pipeline.predict_proba(X_test)[:, 1]
fpr_LR, tpr_LR, _ = roc_curve(y_test, y_pred_LR)
auc_LR = auc(fpr_LR, tpr_LR)

"""Supervised based on random forests"""
Time_RF_Start = time.time()
RF = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
RF.fit(X_train, y_train)
Time_RF_End = time.time()
y_pred_RF = RF.predict_proba(X_test)[:, 1]
fpr_RF, tpr_RF, _ = roc_curve(y_test, y_pred_RF)
auc_RF = auc(fpr_RF, tpr_RF)

print("fpr_rf_lm{}".format(fpr_RF))
print(fpr_RF.shape)
print("tpr_rf_lm{}".format(tpr_RF))
print(tpr_RF.shape)

"""Supervised transformation based on SVM"""
Time_SVM_Start = time.time()
SVC = SVC(C=10, probability=True)
SVC.fit(X_train, y_train)
Time_SVM_End = time.time()
y_pre_svc = SVC.predict_proba(X_test)[:, 1]
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pre_svc)
auc_svc = auc(fpr_svc, tpr_svc)

"""Supervised transformation based on Decision Tree"""
Time_DT_Start = time.time()
DT = DecisionTreeClassifier(max_depth=10)
DT.fit(X_train, y_train)
Time_DT_End = time.time()
y_pre_dt = DT.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pre_dt)
auc_dt = auc(fpr_dt, tpr_dt)

"""Supervised transformation based on MLP"""
Time_MLP_Start = time.time()
MLP = MLPClassifier(hidden_layer_sizes=(40, 40, 10), activation="relu", verbose=True)
MLP.fit(X_train, y_train)
Time_MLP_End = time.time()
y_pre_mlp = MLP.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pre_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)

print("AUC of LR is: %0.3f" % auc_LR)
print("AUC of RF is: %0.3f" % auc_RF)
print("AUC of SVC is: %0.3f" % auc_svc)
print("AUC of DT is:  %0.3f" % auc_dt)
print("AUC of MLP is: %0.3f" % auc_mlp)
print("---------------------")
print("Time of LR Training", Time_LR_End - Time_LR_Start)
print("Time of RF Training", Time_RF_End - Time_RF_Start)
print("Time of SVM Training", Time_SVM_End - Time_SVM_Start)
print("Time of DT Training", Time_DT_End - Time_DT_Start)
print("Time of MLP Training", Time_MLP_End - Time_MLP_Start)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_LR, tpr_LR, label=r'$Logistic\ Regression$')
plt.plot(fpr_RF, tpr_RF, label=r'$Random\ Forest$')
plt.plot(fpr_svc, tpr_svc, label=r'$SVC$')
plt.plot(fpr_dt, tpr_dt, label=r'$Decision\ Tree$')
plt.plot(fpr_mlp, tpr_mlp, label=r'$Neural\ Network$')
plt.xticks(fontsize=10)
plt.xlabel(r'$False\ positive\ rate\ (\%)$', fontsize=15)
plt.ylabel(r'$True\ positive\ rate\ (\%)$', fontsize=15)
plt.title(r'$ROC\ curve$', fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.show()

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


# plt.figure(1,figsize=(20,10))
# ax1=plt.subplot(121)
# ax2=plt.subplot(122)
# tick_spacing = 0.05
# ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# # plt.xticks(np.arange(0, 10, 1.0))
# plt.show()
