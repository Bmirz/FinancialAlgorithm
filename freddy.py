import yfinance as yf

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

sp500.plot.line(y="Close", use_index=True)

del sp500["Dividends"]

del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state = 1)

train = sp500.iloc[:-200]
test = sp500.iloc[-200:]

predictors = ["Open", "Close" , "High", "Low", "Volume"]

model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])

preds = pd.Series(preds, index=test.index)

precision_score(test["Target"], preds, zero_division=(0.0))

combined = pd.concat([test["Target"], preds], axis=1)

combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(sp500,model, predictors)

print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"], predictions["Predictions"]))

print(predictions["Target"].value_counts() / predictions.shape[0])












