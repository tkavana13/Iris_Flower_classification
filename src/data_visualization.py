import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/LENOVO/Iris_Flower_Classification/data/iris.csv')

# Create pairplot
sns.pairplot(data, hue='species')

# Show the plot
plt.show()