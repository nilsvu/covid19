import pandas as pd


def load_jhu_data():
    data = pd.read_csv('data/jhu/time_series_19-covid-Confirmed.csv')
    data.index = data['Country/Region'] + (" (" + data['Province/State'] + ")").fillna("")
    del data["Province/State"]
    del data["Country/Region"]
    del data["Lat"]
    del data["Long"]
    data = data.transpose()
    data.index = pd.to_datetime(data.index)
    return data
