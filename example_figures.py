import plotly.graph_objects as go
import numpy as np

x1 = np.arange(10)

scatter = go.Figure(data=go.Scatter(x=x1, y=x1**2))

x2 = np.random.randn(500)

histogram = go.Figure(data=[go.Histogram(x=x2)])
