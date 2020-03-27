import tensorflow
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Load data from UCI dataset repo
data = np.loadtxt('data_banknote_authentication.txt', delimiter=',')
data = pd.DataFrame(data)

# Add column names
clean_columns = ['variance_of_wavelet', 'skewness_of_wavelet',
                 'curtosis_of_wavelet', 'entropy_of_wavelet',
                 'class']

data.columns = clean_columns

# Separate the target and features as separate dataframes for sklearn APIs
X = data.drop('class', axis=1)
y = data[['class']].astype('int')


# Stratified sampling based on the distribution of the target vector, y
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.20,
                                                    random_state=30)

# Scale features
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Network Architecture
model = Sequential()
model.add(Dense(50,activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train and validate generated model with the unseen test set
model.fit(X_train, y_train.values, epochs=20, validation_data=(X_test, y_test.values))

# Save model objects - predictor and scaler object
model.save('banknote_authentication_model.h5')
joblib.dump(sc, 'scaler.joblib')