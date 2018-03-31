import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('creditcard.csv')
X_raw = data.iloc[:, 0:-1]
y_raw = data.iloc[:, -1]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=0)

# over sampling
sm = SMOTE(random_state=0)
X_train_raw, y_train = sm.fit_sample(X_train_raw, y_train)

# tree based feature selection
tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
tree.fit(X_train_raw, y_train)
model = SelectFromModel(tree, prefit=True)
X_train = model.transform(X_train_raw)
X_test = model.transform(X_test_raw)

seq = np.random.choice(range(X_train.shape[0]), 2000, replace=False)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0][seq][y_train[seq]==0], X_train[:,1][seq][y_train[seq]==0], X_train[:,2][seq][y_train[seq]==0], c = 'b', label = 'class 0')
ax.scatter(X_train[:,0][seq][y_train[seq]==1], X_train[:,1][seq][y_train[seq]==1], X_train[:,2][seq][y_train[seq]==1], c = 'r', label = 'class 1')
ax.legend()
plt.show()