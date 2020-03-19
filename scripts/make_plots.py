import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import locale
import matplotlib.dates as mdates
import logging
import yaml
from scripts.load_data import load_jhu_data


def plot_germany_total(save_to=None, log=False):
    data_jhu = load_jhu_data()

    plt.clf()
    plt.figure(figsize=(10, 6))

    max_timeshift = datetime.timedelta(days=0)
    for timeshift_entry in yaml.safe_load(open('data/timeshifts.yaml', 'r')):
        data_shifted = data_jhu[timeshift_entry['Dataset']].copy()
        timeshift = datetime.timedelta(days=timeshift_entry['ShiftDays'])
        data_shifted.index += timeshift
        data_shifted.plot(marker='.',
                          ls='-',
                          label='{} {} {} Tage'.format(
                              timeshift_entry['Name'],
                              '+' if timeshift.days >= 0 else '-',
                              abs(timeshift.days)),
                          alpha=0.5)
        max_timeshift = max(timeshift, max_timeshift)

    data_jhu['Germany'].plot(marker='.',
                             ls='-',
                             color='black',
                             label='Deutschland')

    plt.xlim(datetime.date(2020, 2, 24), datetime.date.today() + max_timeshift)
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
