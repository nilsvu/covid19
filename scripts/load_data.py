import pandas as pd


def load_jhu_data():
    data = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        index_col=1)
    data.index = data.index + (" (" + data['Province/State'] + ")").fillna("")
    del data["Province/State"]
    del data["Lat"]
    del data["Long"]
    data = data.transpose()
    data.index = pd.to_datetime(data.index)
    return data
