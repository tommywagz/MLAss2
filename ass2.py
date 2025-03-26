from sklearn import datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

# part a
X, y = ds.make_classification(n_samples=1000, n_features=20,
                                  n_informative=5, n_redundant=15,
                                  shuffle=False, random_state=4)

scaler = StandardScaler()
newX = scaler.fit_transform(X)

rand = np.random.default_rng(seed=4)
shuffledIndexes = rand.permutation(newX.shape[0])
shuffledX = newX[shuffledIndexes]
shuffledy = y[shuffledIndexes]

dataClassifier = DecisionTreeClassifier(cirterion="entropy", random_state=4)
dataClassifier.fit(shuffledX, shuffledy)

featureImportance = dataClassifier.feature_importances_

top5Indices = np.argsort(featureImportance)[-5:]
informativeIndices = np.arange(5)

found = 0
for idx in top5Indices:
    if idx in informativeIndices:
        found += 1
        
sortedIndices = np.argsort(featureImportance)[::-1]
sortedImportances = featureImportance[sortedIndices]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), sortedImportances)
plt.xticks(range(X.shape[1]), sortedIndices)
plt.xlabel("Features by Importance")
plt.ylabel("Feature IMportance Score")
plt.title("Feature IMportance from Decision Tree")
plt.show()