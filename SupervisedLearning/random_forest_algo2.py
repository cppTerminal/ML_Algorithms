from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

np.random.seed(0)

iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.head())


df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())

df['is_train'] = np.random.uniform(0,1, len(df)) <= 0.75
print(df.head())

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
print(len(train))
print(len(test))


features = df.columns[:4]
print(features)

y = pd.factorize(train['species'])[0]
print(y)


clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(train[features], y)


x = clf.predict(test[features])
print(x)


x = clf.predict_proba(test[features])
print(x)


preds = iris.target_names[clf.predict(test[features])]
print(preds[0:4])

print(test['species'].head())

#create a confusion matrix
pd.crosstab(test['species'], preds, rownames = ['Actual species'], colnames = ['Predict species'])

preds = iris.target_names[clf.predict(test[features])]
print(preds[0:4])
