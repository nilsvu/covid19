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


def load_cds_data(level='country', upper_levels=()):
    levels = ['country', 'state', 'county', 'city']
    assert level in levels, "Invalid level '{}'. Available: {}".format(level, levels)
    data = pd.read_csv(
        'https://coronadatascraper.com/timeseries.csv',
        index_col=['level'] + levels + ['date'],
        parse_dates=['date'])
    data = data.loc[(level,) + tuple(upper_levels)]
    data.index = data.index.droplevel(levels[len(upper_levels) + 1:])
    return data
