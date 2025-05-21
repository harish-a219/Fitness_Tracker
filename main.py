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
    features = [ 'Age', 'Height (cm)', 'Weight (kg)','Heart Rate (bpm)', 'Steps Taken', 'Distance (km)','Daily Calories Intake']
    target = 'Calories Burned'

    normalized_data = normalize(df[features])
    df_normalized = pd.DataFrame(normalized_data, columns = features)
    X = df_normalized
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size = 0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    
    print(y_predict)
    return model

def main():
    data = importData()
    modelSetup(data)
main()