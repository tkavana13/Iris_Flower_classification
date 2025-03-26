import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    data = load_data('C:/Users/LENOVO/Iris_Flower_Classification/data/iris.csv')
    print(data.head())