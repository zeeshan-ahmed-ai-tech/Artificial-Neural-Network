import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('placement.csv')

print(df.shape)
df.head()

sns.scatterplot(x='cgpa', y='resume_score', hue='placed', data=df)

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

from sklearn.linear_model import Perceptron
p = Perceptron()

p.fit(X,y)

p.coef_

p.intercept_

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X.values, y.values, clf=p, legend=2)