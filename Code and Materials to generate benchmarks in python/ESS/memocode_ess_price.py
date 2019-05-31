import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from Utils.csvread import CsvReader
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
from pandas import DataFrame


Data_f = {'data1': [1, 1],'data2': [0, 1], 'data3': [1, 0], 'data4': [1, 1],'data5': [0, 1], 'data6': [1, 0], 'data7': [1, 1],'data8': [0, 1], 'data9': [1, 0], 'data10': [1, 0]}
features = DataFrame (Data_f, columns = ['data1','data2', 'data3', 'data4','data5', 'data6', 'data7','data8', 'data9', 'data10'])

Data_l = {'output1': [0, 0]}
labels = DataFrame (Data_l, columns = ['output1'])

model = Sequential()
model.add(Dense(5, input_dim=10, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.fit(features, labels)

# serialize model to JSON
model_json = model.to_json()
with open("memocode_ess_price.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("memocode_ess_price.h5")
print("Saved model to disk")
