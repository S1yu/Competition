import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

train = pd.read_csv("./store-sales-time-series-forecasting/train.csv")
test = pd.read_csv("./store-sales-time-series-forecasting/test.csv")
stores = pd.read_csv("./store-sales-time-series-forecasting/stores.csv")
transactions = pd.read_csv("./store-sales-time-series-forecasting/transactions.csv").sort_values(["store_nbr", "date"])

train["date"]=pd.to_datetime(train.date)
test["date"]=pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)
# data type
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

train = train.head(1000)
# print groupby .groups
train.groupby(["date","store_nbr"])

