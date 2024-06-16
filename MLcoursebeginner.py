import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

scaler = StandardScaler()

data = pd.read_csv("housing.csv")

data.dropna(inplace=True)


x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

train_data = x_train.join(y_train)
test_data = x_test.join(y_test)

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']


test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

x_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

print(test_data)
print(train_data)

#Scatterplot:
#sns.scatterplot(x='latitude', y='longitude', data=train_data , hue='median_house_value', palette='coolwarm')

#train_data.hist()


#Heat map:
#sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

reg = LinearRegression()

reg.fit(x_train, y_train)

LinearRegression()

print(reg.score(x_train, y_train))

reg.fit(x_test, y_test)

LinearRegression()

print(reg.score(x_test, y_test))


#scsle it
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.fit_transform(x_test)

#Forest
forest = RandomForestRegressor()

#forest.fit(x_train, y_train)
#print(forest.score(x_train, y_train))

#forest.fit(x_test, y_test)
#print(forest.score(x_test, y_test))

forest.fit(x_test_s, y_test)
print(forest.score(x_test_s, y_test))

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [30, 50, 100],
    "max_features": [8, 12, 20],
    "min_samples_split": [2, 4, 6, 8]
    }

grid_search = GridSearchCV(forest, param_grid, cv=5,
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True)

grid_search.fit(x_train_s, y_train)


best_forest = grid_search.best_estimator_

print(best_forest.score(x_test_s, y_test))