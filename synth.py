import random

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))


class Synth():

	def __init__(self):
		print("synth init")

	def LogisticRegressionScore(self, df, columns):
		train = df[df['split'] == 'train']
		test = df[df['split'] == 'test']

		y_train = train['label'].values
		X_train = train[columns].values

		model = LogisticRegression()
		model.fit(X_train, y_train)

		X = df[columns].values
		y_score = model.predict_proba(X)[:, 1]

		return y_score

	def runModel(self):
		years = list(range(2010, 2020))
		mu = [0.0, 0.0, 0.0, 0.0]
		cov = [[1.0, 0.2, 0.0, -0.1],
		       [0.2, 1.0, 0.0, 0.0],
		       [0.0, 0.0, 1.0, 0.0],
		       [-0.1, 0.0, 0.0, 1.0]]
		N = 200 + random.randint(100, 600)
		print('N is ', N)

		df_list = []
		for year in years:
			X = np.random.multivariate_normal(mu, cov, N)
			df = pd.DataFrame(X)
			df['male'] = np.random.choice([0, 1], N)

			W = [2.5, -1.5, 0.5, 1.0, 0.0]
			b = -3.5
			e = np.random.normal(0.0, 3.5, N)
			err = np.random.normal(0.0, 0.1, len(W))

			df['raw'] = sigmoid(np.matmul(df.values, W + err) + b + e)
			df['label'] = 0
			df.loc[df['raw'] > 0.9, 'label'] = 1.0

			df['year'] = year
			df['gender'] = 'M'
			df.loc[df['male'] == 0, 'gender'] = 'F'

			df_list.append(df)

		df = pd.concat(df_list)
		df.reset_index(inplace=True, drop=True)
		df.drop(columns='raw', inplace=True)
		df['split'] = 'train'
		df.loc[df['year'] > 2016, 'split'] = 'test'

		df['clf_1'] = self.LogisticRegressionScore(df, [0, 'male'])
		df['clf_2'] = self.LogisticRegressionScore(df, [0, 1, 'male'])
		df['clf_3'] = self.LogisticRegressionScore(df, [0, 1, 2, 'male'])
		df['clf_4'] = self.LogisticRegressionScore(df, [0, 1, 2, 3, 'male'])

		return df
