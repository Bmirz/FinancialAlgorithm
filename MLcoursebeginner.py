import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("housing.csv")

data.dropna(inplace=True)


x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, y_trian, y_train, y_test = train_test_split(x,y, test_size=0.2)

train_data = x_train.join(y_train)

sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

train_data.hist()

plt.show()


