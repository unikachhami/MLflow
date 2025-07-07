import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')




wine = load_wine()
x = wine.data
y = wine.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

n_estimators = 10
max_depth = 5


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)

    

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    
    plt.savefig("Confusion Matrix.png")

    mlflow.log_artifact("Confusion Matrix.png")
    mlflow.log_artifact(__file__)
    
    mlflow.set_tags({'Authur':'Unik','Project':'Wines'})

    mlflow.sklearn.log_model(rf,'RandomForestClassifier')


    print(accuracy)