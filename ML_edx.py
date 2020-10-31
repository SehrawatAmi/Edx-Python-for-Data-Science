## Classification - Decision Tree Approach

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

### data could be downloaded from Edx course Python For Data Science Week 7 materials
data = pd.read_csv('D:\Python Projects\Python for Data Science_UCSanDiego_edx\Week-7-MachineLearning\weather\daily_weather.csv')

print (data.head())

print (data.columns)

print (data[data.isnull().any(axis=1)])

del data['number']

before_rows = data.shape[0]
print (before_rows)

data = data.dropna()

after_rows = data.shape[0]
print(after_rows)

## How many rows dropped due to Cleaning?
print ('na rows dropped; ' + str(before_rows - after_rows))

## Convert to a Classification Task
clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm']> 24.99)* 1
##print (clean_data['high_humidity_label'])

## Target is stored in 'y'
y = clean_data[['high_humidity_label']].copy()
##print (y)
##print (clean_data['relative_humidity_3pm'].head())
##print (y.head())

## Use 9am Sensor Signals as Features to Predict Humidity at 3 pm
morning_features = ['air_pressure_9am', 'air_temp_9am','avg_wind_direction_9am','max_wind_direction_9am','max_wind_speed_9am'
            ,'rain_accumulation_9am', 'rain_duration_9am']
x = clean_data[morning_features].copy()
##print (x.columns)
##print (y.columns)

#### Perform Test and Train Split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=324)

print (type(X_train))
print (type(X_test))
print (type(Y_train))
print (type(Y_test))

print (X_train.head())
print (Y_train.describe())

## Fit on Train Set
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
print (humidity_classifier.fit(X_train,Y_train))

print (type(humidity_classifier))

## Predict on Test Set
predictions = humidity_classifier.predict(X_test)


## To check the prediction number vs the Actual Number
print (predictions[:10])
print (Y_test['high_humidity_label'][:10])

## To check the accuracy of the model between predicted value vs actual value
print (accuracy_score(y_true=  Y_test, y_pred=predictions))