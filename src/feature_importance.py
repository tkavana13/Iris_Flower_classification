import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.show()

if __name__ == "__main__":
    model = joblib.load('iris_model.pkl')
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    plot_feature_importance(model, feature_names)