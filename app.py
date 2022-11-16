import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

import example_figures as example

from synth import Synth
from roc import Roc

# model = Synth()
# result = model.runModel()
# result.to_csv('synth.csv')
# print(result)


figs = {
	'ROC Curves': '',
	'Example Scatter': example.scatter,
	'Example Histogram': example.histogram
}
#
app = dash.Dash()
#
# #fig_names = ['scatter', 'histogram']
fig_names = list(figs.keys())


fig_button = html.Div([
	html.Button(id='submit-button-state', n_clicks=0, children='Run Model')
])
# fig_dropdown = html.Div([
#     dcc.Dropdown(
#         id='fig_dropdown',
#         options=[{'label': x, 'value': x} for x in fig_names],
#         value=None
#     )])
fig_plot = html.Div(id='fig_plot')
fig_plot2 = html.Div(id='fig_plot2')
app.layout = html.Div([fig_button, fig_plot, fig_plot2])

@app.callback(
dash.dependencies.Output('fig_plot2', 'children'),
[dash.dependencies.Input('submit-button-state', 'n_clicks')])

def update_output(fig_name):
	model = Synth()
	result = model.runModel()
	# result.to_csv('synth.csv')

	# print("new data")
	# print(result.head())

	return dash_table.DataTable(result.to_dict('records'))

@app.callback(
dash.dependencies.Output('fig_plot', 'children'),
[dash.dependencies.Input('submit-button-state', 'n_clicks')])

def update_output(fig_name):
	model = Synth()
	result = model.runModel()
	result.to_csv('synth.csv')

	print("new data")
	print(result.head())

	# return dash_table.DataTable(result.to_dict('records'))
	return name_to_figure(fig_name)

def name_to_figure(fig_name):
	fig_name = 'ROC Curves'
	figure = go.Figure()
	for name in fig_names:
		if fig_name == name:
			figure = figs[name]

	roc = Roc()
	currFig = roc.getFig()
	figs['ROC Curves'] = currFig
	print("train finish")
	return dcc.Graph(figure=figure)

app.run_server(debug=True, use_reloader=False, port=8000)
