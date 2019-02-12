import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

print(__doc__)

dataset = pd.read_csv("F:\PyCharm_Project/201810_ML_Learning\All_Data/Sum_OF_All_Data_KMJ1_7.csv")
X = dataset.iloc[:, 0:17].values
y = dataset.iloc[:, 18].values

# The scorers can be either be one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# Setting refit='AUC', refits an estimator on the whole dataset with the
# parameter setting that has the best cross-validated AUC score.
# That estimator is made available at ``gs.best_estimator_`` along with
# parameters like ``gs.best_score_``, ``gs.best_params_`` and
# ``gs.best_index_``

"""注意一定要含有交叉验证"""

gs = GridSearchCV(DecisionTreeClassifier(splitter="best",random_state=None,max_features="auto"),
                  param_grid={'min_samples_split': np.arange(0.01,1,0.1),'max_depth': np.arange(1,10,1)},
                  scoring=scoring, cv=5, refit='AUC', return_train_score=True)
gs.fit(X, y)

print("best_params:", gs.best_params_)
results = gs.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("Internal_Node_Split_Fraction")
plt.ylabel("Score")

ax = plt.gca()
# ax.set_xlim(0, 402)
ax.set_ylim(0.2, 1)
xmajorLocator= MultipleLocator(0.1)
xmajorFormatter = FormatStrFormatter('%0.1f')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
print("Cross_validation_results:",results)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

# print(sorted())

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()