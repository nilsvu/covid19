import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import locale
import matplotlib.dates as mdates
import logging


def plot_germany_total(save_to=None, log=False):
    data = pd.read_csv(
        'data/jhu/time_series_19-covid-Confirmed.csv'
    ).set_index('Country/Region')
    plt.clf()

    data_de = data.loc['Germany'].iloc[3:]
    data_de.index = pd.to_datetime(data_de.index)

    data_it = data.loc['Italy'].iloc[3:]
    data_it.index = pd.to_datetime(data_it.index)

    data_it_shifted = data_it.copy()
    it_shift = datetime.timedelta(days=9)
    data_it_shifted.index += it_shift

    data_it_shifted.plot(marker='.',
                         color='red',
                         ls='-',
                         label='Italien + {} Tage'.format(it_shift.days),
                         alpha=0.5)
    data_de.plot(marker='.',
                 color='black',
                 ls='-',
                 figsize=(10, 6),
                 label="Deutschland")
    plt.xlim(datetime.date(2020, 2, 24), datetime.date.today() + it_shift)
    plt.axvline(datetime.date.today(), color='black', alpha=0.2, lw=2)
    plt.title("COVID-19 Infektionen")
    plt.legend()

    plt.grid(color=(0.9, 0.9, 0.9), lw=1, axis='y')

    if log:
        plt.yscale('log')

    if save_to:
        plt.savefig(save_to)


def plot_states(save_to=None, log=False):
    plt.clf()

    data_rki = pd.read_csv('data/rki/states_timeseries.csv').set_index(
        'Date')
    data_rki.index = pd.to_datetime(data_rki.index, dayfirst=True)

    plt.figure(figsize=(10, 6))
    plt.title("COVID-19 Infektionen pro Bundesland")
    for state in data_rki.columns[np.argsort(data_rki.iloc[-1])[::-1]]:
        data_rki[state].plot(marker='.', ls='-')
        plt.annotate(state, (data_rki.index[-1], data_rki[state][-1]),
                     textcoords='offset points',
                     xytext=(8, 0),
                     bbox=dict(fc=(1, 1, 1, 0.5), ec=(1, 1, 1, 0), pad=2))
    plt.xlabel(None)
    plt.xlim(data_rki.index[0],
             datetime.date.today() + datetime.timedelta(days=1))
    plt.axvline(datetime.date.today(), alpha=0.2, color='black', lw=3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.grid(color=(0.9, 0.9, 0.9), lw=1, axis='y')

    if log:
        plt.yscale('log')

    if save_to:
        plt.savefig(save_to)


if __name__ == "__main__":
    try:
        locale.setlocale(locale.LC_TIME, "de_DE")
    except locale.Error:
        logging.warning("Unable to set locale")

    import os
    os.makedirs('plots', exist_ok=True)
    plot_germany_total('plots/germany_total.svg')
    plot_germany_total('plots/germany_total_log.svg', log=True)
    plot_states('plots/states.svg')
    plot_states('plots/states_log.svg', log=True)
