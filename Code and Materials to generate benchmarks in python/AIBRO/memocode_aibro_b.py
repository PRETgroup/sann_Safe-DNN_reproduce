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


Data_f = {'data1': [1, 1],'data2': [0, 1], 'data3': [1, 0], 'data4':[1, 0]}
features = DataFrame (Data_f, columns = ['data1','data2', 'data3', 'data4'])

Data_l = {'output1': [0, 0], 'output2': [1, 0]}
labels = DataFrame (Data_l, columns = ['output1', 'output2'])

model = Sequential()
model.add(Dense(3, input_dim=4, activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.fit(features, labels)

# serialize model to JSON
model_json = model.to_json()
with open("memocode_aibro_b.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("memocode_aibro_b.h5")
print("Saved model to disk")
