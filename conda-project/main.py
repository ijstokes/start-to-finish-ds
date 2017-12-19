#!/usr/bin/env python
'''
A pre-trained fuel efficiency model that takes inputs `cyl displ hp weight accel yr`
and returns the MPG fuel efficiency.

Trained on a subset of the 1970-1982 `autompg` data set [1]

(C) Anaconda Inc, Ian Stokes-Rees, September 2017

[1] https://archive.ics.uci.edu/ml/datasets/auto+mpg
'''

import pandas as pd

from sklearn.externals.joblib import load

from bokeh.charts import Scatter
from bokeh.io import curdoc, show, output_notebook, output_file

mpg_model = load('mpg_linear_regression_model.pkl')
data      = pd.read_csv('mpg_data.csv')

# augment the mpg DataFrame with the prediction
data['prediction'] = mpg_model.predict(data['cyl displ hp weight accel yr'.split()])

s1 = Scatter(data=data,
            x='mpg', y='prediction', color='origin',
            height=300, width=600,
            title='Fuel efficiency predictions of selected vehicles from 1970-1982',
            tools='hover, box_zoom, lasso_select, save, reset',
            tooltips = [
              ('model','@name'),
              ('HP',  '@hp'),
              ('actual MPG', '@mpg'),
              ('predicted MPG', '@prediction')
            ])

s2 = Scatter(data=data,
            x='yr', y='mpg', color='origin',
            height=300, width=600,
            title='Fuel efficiency of selected vehicles from 1970-1982',
            tools='hover, box_zoom, save, reset',
            tooltips = [
              ('model','@name'),
              ('HP',  '@hp'),
              ('cyl', '@cyl'),
              ('weight', '@weight')
            ])

curdoc().add_root(s1)
curdoc().add_root(s2)
curdoc().title = "MPG Prediction"
