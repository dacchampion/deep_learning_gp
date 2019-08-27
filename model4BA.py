import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D

np.random.seed(2)

NUM_SAMPLES = 36
BATCH_SIZE = 128  # Number of samples for training
NUM_CLASSES = 1  # Lying or not
INPUT_VARIABLES = 36  # Affectiva plus xLabs output variables
SAMPLES_PER_SECOND = 5  # samples per second in 30 seconds
SECONDS_PER_PERSON = 30
TRAIN_EPOCHS = 50

NUM_HIDDEN_LSTM = 32
NUM_FEAT_MAP = 16


def conv_lstm():
    model = Sequential()
    model.add(Conv1D(NUM_FEAT_MAP, input_shape=(SAMPLES_PER_SECOND, SECONDS_PER_PERSON), kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(NUM_FEAT_MAP, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    #model.add(Permute((2, 1, 3)))  # for swap-dimension
    #model.add(Reshape((-1, SAMPLES_PER_SECOND, SECONDS_PER_PERSON)))
    model.add(LSTM(NUM_HIDDEN_LSTM, return_sequences=False, stateful=False))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


Event = pd.read_csv('EventTable.csv')
Expression = pd.read_csv('ExpressionTable.csv')
Gaze = pd.read_csv('GazeTable.csv')
session_question = pd.read_csv('sessionQuestion.csv')
session = pd.read_csv('sessionTable.csv')

x_df1 = Expression.iloc[:, 7:]
x_df1['sessionQuesID'] = Expression.iloc[:, 1]
No_quest = np.max(x_df1['sessionQuesID'])
features = x_df1.shape[1] - 1
X = []
n = 5
for i in range(No_quest):
    x1 = x_df1[x_df1['sessionQuesID'] == i + 1]
    x1 = x1.drop(['sessionQuesID'], axis=1)
    No_data = int(x1.shape[0] / n)
    a = 0
    for j in range(n):
        X.extend(np.mean(x1.iloc[a:a + No_data, :], axis=0).values.tolist())
        a = a + No_data

X = np.reshape(X, (No_quest, n * features))
X_AUG = np.reshape(X, (NUM_SAMPLES, SAMPLES_PER_SECOND, SECONDS_PER_PERSON))

Y = session_question.iloc[1:, 4].values.tolist()

predictor = conv_lstm()
H = predictor.fit(X_AUG, Y,
                  batch_size=BATCH_SIZE,
                  epochs=TRAIN_EPOCHS,
                  verbose=1,
                  validation_split=0.2)