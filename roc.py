import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc, roc_auc_score

class Roc():

	def roc_plot_data(self, df, label='label', score='score', split_column=None):
		if split_column:

			plot_df_list = []

			for level in df[split_column].unique():
				sub = df[df[split_column] == level]
				y = sub[label]
				y_score = sub[score]
				fpr, tpr, threshold = roc_curve(y, y_score)

				plot_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr,
				                        'threshold': threshold})
				plot_df['partial_auc'] = np.nan
				plot_df[split_column] = level

				partial_auc = plot_df.loc[plot_df['fpr'] > 0.0, 'fpr'].apply(lambda x: roc_auc_score(y, y_score, max_fpr=x))
				plot_df.loc[plot_df['fpr'] > 0.0, 'partial_auc'] = partial_auc

				plot_df_list.append(plot_df.copy())

			plot_df = pd.concat(plot_df_list)

		else:

			y = df[label]
			y_score = df[score]
			fpr, tpr, threshold = roc_curve(y, y_score)

			plot_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr,
			                        'threshold': threshold})
			plot_df['partial_auc'] = np.nan

			partial_auc = plot_df.loc[plot_df['fpr'] > 0.0, 'fpr'].apply(lambda x: roc_auc_score(y, y_score, max_fpr=x))
			plot_df.loc[plot_df['fpr'] > 0.0, 'partial_auc'] = partial_auc

		return plot_df


	def make_visible_dict(self, plot_dfs_list, split_cols, version, plot_dfs):
		visible_true = {}
		visible_false = {}

		for split_col in split_cols:
			plot_df = plot_dfs[version][split_col]
			true_list = [True] * len(plot_df[split_col].unique())
			false_list = [False] * len(plot_df[split_col].unique())
			visible_true[split_col] = true_list.copy()
			visible_false[split_col] = false_list.copy()

		visible_dict = {}
		for split_col in split_cols:
			visible_dict[split_col] = []
			for key in split_cols:
				if key == split_col:
					visible_dict[split_col].extend(visible_true[key])
				else:
					visible_dict[split_col].extend(visible_false[key])

		return visible_dict

	def getFig(self):
		print('roc_fig reload csv')
		df = pd.read_csv('synth.csv')

		df2 = df[['label', 'year', 'gender', 'split', 'clf_1', 'clf_2', 'clf_3', 'clf_4']].copy()
		df2 = pd.melt(df2, id_vars=['label', 'year', 'gender', 'split'],
		              value_vars=['clf_1', 'clf_2', 'clf_3', 'clf_4'],
		              var_name='model', value_name='score')

		versions = ['clf_3', 'clf_4']
		split_cols = ['gender', 'split', 'year']

		print('getFig 2')
		plot_dfs = {}
		for version in versions:
			plot_dfs[version] = {}
			for split_col in split_cols:
				plot_dfs[version][split_col] = self.roc_plot_data(df2[df2['model'] == version], split_column=split_col)
				plot_dfs[version][split_col] = plot_dfs[version][split_col].round(4)

		version = versions[0]
		visible = self.make_visible_dict(plot_dfs, split_cols, version, plot_dfs)
		title = str(version) + ' ROC Curves'

		print('getFig 3')
		layout = go.Layout(
			title=title,
			yaxis=dict(title='True Positive Rate'),
			xaxis=dict(title='False Positive Rate'),
			template='plotly'
		)

		print('getFig 4')
		fig = go.Figure(layout=layout)

		print('getFig 5')
		fig.add_shape(
			type='line', line=dict(dash='dash'),
			x0=0, x1=1, y0=0, y1=1
		)

		print('getFig 6')
		for split_col in split_cols:

			plot_df = plot_dfs[version][split_col]

			for level in plot_df[split_col].unique():
				fpr = plot_df[plot_df[split_col] == level]['fpr']
				tpr = plot_df[plot_df[split_col] == level]['tpr']
				threshold = plot_df[plot_df[split_col] == level]['threshold']
				partial_auc = plot_df[plot_df[split_col] == level]['partial_auc']
				auc = partial_auc.max()

				customdata = np.stack((threshold, partial_auc), axis=-1)

				hovertemplate = '<br><b>TPR</b>: %{y}<br>'
				hovertemplate += '<b>Threshold</b>: %{customdata[0]}'
				hovertemplate += '<br><b>Partial AUC</b>: %{customdata[1]}<br>'

				fig.add_trace(go.Scatter(x=fpr, y=tpr, customdata=customdata,
				                         hovertemplate=hovertemplate, mode='lines',
				                         name=str(level) + f' (AUC={auc})',
				                         visible=False))

		button_layer_1_height = 1.22
		button_shift = 0.3

		print('getFig 7')
		annotations = [dict(text="Subset", x=button_shift - 0.02, xref='paper', xanchor='right',
		                    y=button_layer_1_height - 0.02, yref='paper', yanchor='top', align="right", showarrow=False)]

		fig.update_layout(hovermode='x unified',
		                  updatemenus=[
			                  dict(
				                  type='buttons',
				                  direction='right',
				                  active=-1,
				                  x=button_shift,
				                  xanchor="left",
				                  y=button_layer_1_height,
				                  buttons=list([
					                  dict(label=split_col,
					                       method='update',
					                       args=[{'visible': visible[split_col]},
					                             {'title': title,
					                              'annotations': annotations}])
					                  for split_col in split_cols
				                  ])
			                  )
		                  ],
		                  annotations=annotations
		                  )

		return fig

