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


Data_f = {'data1': [1, 1],'data2': [0, 1], 'data3': [1, 0], 'data4': [1, 1],'data5': [0, 1], 'data6': [1, 0], 'data7': [1, 1],'data8': [0, 1], 'data9': [1, 0], 'data10': [1, 0], 'data11': [1, 1],'data12': [0, 1], 'data13': [1, 0], 'data14': [1, 1],'data15': [0, 1], 'data16': [1, 0], 'data17': [1, 1],'data18': [0, 1], 'data19': [1, 0], 'data20': [1, 0], 'data21': [1, 0]}
features = DataFrame (Data_f, columns = ['data1','data2', 'data3', 'data4','data5', 'data6', 'data7','data8', 'data9', 'data10', 'data11','data12', 'data13', 'data14','data15', 'data16', 'data17','data18', 'data19', 'data20', 'data21'])

Data_l = {'data1': [1, 1],'data2': [0, 1], 'data3': [1, 0], 'data4': [1, 1],'data5': [0, 1], 'data6': [1, 0], 'data7': [1, 1],'data8': [0, 1], 'data9': [1, 0], 'data10': [1, 0], 'data11': [1, 1],'data12': [0, 1], 'data13': [1, 0], 'data14': [1, 1],'data15': [0, 1], 'data16': [1, 0], 'data17': [1, 1],'data18': [0, 1], 'data19': [1, 0], 'data20': [1, 0], 'data21': [1, 0], 'data22': [1, 1],'data23': [0, 1], 'data24': [1, 0], 'data25': [1, 1],'data26': [0, 1], 'data27': [1, 0], 'data28': [1, 1],'data29': [0, 1], 'data30': [1, 0], 'data31': [1, 0], 'data32': [1, 1],'data33': [0, 1], 'data34': [1, 0], 'data35': [1, 1],'data36': [0, 1], 'data37': [1, 0], 'data38': [1, 1],'data39': [0, 1], 'data40': [1, 0], 'data41': [1, 0]}
labels = DataFrame (Data_l, columns = ['data1','data2', 'data3', 'data4','data5', 'data6', 'data7','data8', 'data9', 'data10', 'data11','data12', 'data13', 'data14','data15', 'data16', 'data17','data18', 'data19', 'data20', 'data21', 'data22','data23', 'data24', 'data25','data26', 'data27', 'data28','data29', 'data30', 'data31', 'data32','data33', 'data34', 'data35','data36', 'data37', 'data38','data39', 'data40', 'data41'])

model = Sequential()
model.add(Dense(10, input_dim=21, activation='softmax'))
model.add(Dense(41, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.fit(features, labels)

# serialize model to JSON
model_json = model.to_json()
with open("memocode_ess_q.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("memocode_ess_q.h5")
print("Saved model to disk")
