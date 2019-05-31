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


csvreader = CsvReader()
data = csvreader.read('memocode_adder_inputs.csv')
labels = data[['output']].copy()
features = data[['data1', 'data2']].copy()

Data_f = {'data1': [10000, 30000],'data2': [20000, -5000]}
test_features = DataFrame (Data_f, columns = ['data1','data2'])

Data_l = {'output': [26897, 27837]}
test_labels = DataFrame (Data_l, columns = ['output'])

model = Sequential()
model.add(Dense(5, input_dim=2, activation='linear',use_bias=False))
model.add(Dense(1, activation='linear',use_bias=False))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.fit(features, labels)

# serialize model to JSON
model_json = model.to_json()
with open("memocode_adder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("memocode_adder.h5")
print("Saved model to disk")

print(test_features)
predictions=model.predict(test_features)
print(predictions)
