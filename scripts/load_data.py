import pandas as pd


def apply_data_corrections(dataframe_confirmed_DF):
    # https://github.com/CSSEGISandData/COVID-19/issues/833
    confirmed_fixes_dict = {
        'Italy|3/12/20': 15113,
        'Spain|3/12/20': 3146,
        'France|3/12/20': 2876,
        'United Kingdom|3/12/20': 590,
        'Germany|3/12/20': 2745,
        'Argentina|3/12/20': 19,
        'Belgium|3/12/20': 314,
        'Chile|3/12/20': 23,
        'Greece|3/12/20': 98,
        'Indonesia|3/12/20': 34,
        'Ireland|3/12/20': 43,
        'Japan|3/12/20': 620,
        'Netherlands|3/12/20': 503,
        'Qatar|3/12/20': 262,
        'Singapore|3/12/20': 178,
        'United Kingdom|3/12/20': 1391,
        'France|3/15/20': 5423
    }
    for key in confirmed_fixes_dict.keys():
        country_to_be_fixed = key.split('|')[0]
        date_to_be_fixed = key.split('|')[1]
        value_to_be_fixed = confirmed_fixes_dict[key]
        dataframe_confirmed_DF.at[country_to_be_fixed,
                                  date_to_be_fixed] = value_to_be_fixed
    return dataframe_confirmed_DF


def load_jhu_data():
    data = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
        index_col=1)
    data = apply_data_corrections(data)
    data.index = data.index + (" (" + data['Province/State'] + ")").fillna("")
    del data["Province/State"]
    del data["Lat"]
    del data["Long"]
    data = data.transpose()
    data.index = pd.to_datetime(data.index)
    return data
