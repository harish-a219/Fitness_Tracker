import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def importData():
    df = pd.read_csv('workout_fitness_tracker_data.csv')
    return df

def modelSetup(df):
    features = ['Age', 'Height (cm)', 'Weight (kg)','Heart Rate (bpm)', 'Steps Taken', 'Distance (km)','Daily Calories Intake']
    target = 'Calories Burned'

    normalized_data = normalize(df[features])
    df_normalized = pd.DataFrame(normalized_data, columns = features)
    X = df_normalized
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size = 0.2)
    model = LogisticRegression()
    model.fit(X_train.values, y_train)

    y_predict = model.predict(X_test)
    
    print(y_predict)
    return model, features

def main():
    data = importData()
    data1, data2 = modelSetup(data)
    user_inputs = userInputs()
    print(predictions(user_inputs, data2, data1))


def userInputs():
    user = {}
    age = int(input('What is your age?'))
    user['Age'] = age

    height = int(input('What is your height in cm?'))
    user['Height (cm)'] = height

    weight = int(input('What is your weight in kg?'))
    user['Weight (kg)'] = weight

    heart_rate = int(input('What is your heart rate in BPM?'))
    user['Heart Rate (bpm)'] = heart_rate

    steps = int(input('How many steps do you take per day on average?'))
    user['Steps Taken'] = steps

    distance = int(input('How much do you walk per day in km?'))
    user['Distance (km)'] = distance

    calorie = int(input('What is your daily calorie intake on average?'))
    user['Daily Calories Intake'] = calorie

    return user


def predictions(user_inputs, features, model):
    num_values = []
    for i in features:
        num_values.append(user_inputs[i])
    
    num_values = normalize([num_values])[0]

    
    final = num_values.reshape(1,-1)   #Predict function only takes 2-D arrays, so final.reshape makes the list go from 1-D to 2-D
    prediction = model.predict(final)[0]
    probability = model.predict_proba(final)
    return 'Probability:'+ str(round(probability[0][1]*100,2)) + '% chance'

main()