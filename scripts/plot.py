import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yaml
import numpy as np
import pandas as pd
from .analyze import fit_model, get_logistic_date_end, logistic, logistic_deriv
from .load_data import load_jhu_data


def plot_timeshifts(save_to=None, log=False):
    data = load_jhu_data()

    plt.clf()
    plt.figure(figsize=(10, 6))

    max_timeshift = datetime.timedelta(days=0)
    for timeshift_entry in yaml.safe_load(open('data/timeshifts.yaml', 'r')):
        data_shifted = data[timeshift_entry['Dataset']].copy()
        timeshift = datetime.timedelta(days=timeshift_entry['ShiftDays'])
        data_shifted.index += timeshift
        data_shifted.plot(marker='.',
                          ls='-',
                          label='{} {} {} Tage'.format(
                              timeshift_entry['Name'],
                              '+' if timeshift.days >= 0 else '-',
                              abs(timeshift.days)),
                          alpha=0.5)
        plt.scatter(data_shifted.index[-1],
                    data_shifted.values[-1],
                    marker='o')
        max_timeshift = max(timeshift, max_timeshift)

    data['Germany'].plot(marker='.',
                         ls='-',
                         color='black',
                         label='Deutschland')

    plt.xlim(datetime.date(2020, 2, 24), datetime.date.today() + max_timeshift)
    plt.axvline(datetime.date.today(), color='black', alpha=0.2, lw=2)
    plt.title("Zeitversatz in Europa")
    plt.ylabel("Ansteckungen Gesamt")
    plt.legend()

    plt.grid(color=(0.9, 0.9, 0.9), lw=1, axis='y')

    if log:
        plt.yscale('log')

    if save_to:
        plt.savefig(save_to)


def plot_german_states(save_to=None, log=False):
    data_rki = pd.read_csv('data/rki/states_timeseries.csv').set_index('Date')
    data_rki.index = pd.to_datetime(data_rki.index, dayfirst=True)

    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.title("Ansteckungen pro Bundesland in Deutschland")
    for state in data_rki.columns[np.argsort(data_rki.iloc[-1])[::-1]]:
        data_rki[state].plot(marker='.', ls='-')
        plt.annotate(state, (data_rki.index[-1], data_rki[state][-1]),
                     textcoords='offset points',
                     xytext=(8, 0),
                     bbox=dict(fc=(1, 1, 1, 0.5), ec=(1, 1, 1, 0), pad=2))
    plt.xlabel(None)
    plt.ylabel("Ansteckungen Gesamt")
    plt.xlim(data_rki.index[0],
             datetime.date.today() + datetime.timedelta(days=3))
    plt.axvline(datetime.date.today(), alpha=0.2, color='black', lw=3)

    plt.grid(color=(0.9, 0.9, 0.9), lw=1, axis='y')

    if log:
        plt.yscale('log')

    if save_to:
        plt.savefig(save_to)


def plot_prediction(data, label, *args, save_to=None, **kwargs):
    popt, sigma, t_fit_first = fit_model(data, *args, **kwargs)
    popt_err, sigma_err, _ = fit_model(data[:-1], *args, **kwargs)
    t_end, date_end = get_logistic_date_end(popt)
    _, date_end_lag = get_logistic_date_end(popt_err)
    date_end_err = abs(date_end_lag - date_end)
    date_end_conservative = max(date_end, date_end + date_end_err)

    plt.figure(figsize=(10, 6))

    plot_end_date = date_end_conservative + datetime.timedelta(days=3)

    t_prognosis = np.arange(t_fit_first,
                            (plot_end_date - datetime.date.today()).days)
    t_prognosis_dates = [
        datetime.date.today() + datetime.timedelta(days=int(t))
        for t in t_prognosis
    ]
    plt.plot(t_prognosis_dates,
             logistic_deriv(t_prognosis, *popt_err),
             label="Prognose gestern: {:%-d. %B %Y}".format(date_end_lag))
    plt.fill_between(t_prognosis_dates,
                     logistic_deriv(t_prognosis, popt_err[0], popt_err[1],
                                    popt_err[2] + sigma_err[2]),
                     logistic_deriv(t_prognosis, popt_err[0], popt_err[1],
                                    popt_err[2] - sigma_err[2]),
                     alpha=0.3)
    plt.plot(t_prognosis_dates,
             logistic_deriv(t_prognosis, *popt),
             label="Prognose heute: {:%-d. %B %Y}".format(date_end))
    plt.fill_between(t_prognosis_dates,
                     logistic_deriv(t_prognosis, popt[0], popt[1],
                                    popt[2] + sigma[2]),
                     logistic_deriv(t_prognosis, popt[0], popt[1],
                                    popt[2] - sigma[2]),
                     alpha=0.3)

    plt.ylabel('COVID-19 Neuansteckungen')
    plt.xlim(data.index[0], plot_end_date)
    plt.axvline(datetime.date.today(), color='black', alpha=0.2, lw=2)
    plt.axvline(
        date_end_conservative,
        color='black',
        lw=1,
        ls='dashed',
        label="Pessimistisch: {:%-d. %B %Y}".format(date_end_conservative))
    plt.axvline(datetime.date.today() +
                datetime.timedelta(days=int(t_fit_first)),
                color='black',
                lw=1,
                ls='dotted')

    plt.plot(data.index[1:],
             np.diff(data),
             marker='.',
             ls='-',
             color='black',
             label='Daten ({})'.format(label))

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    plt.title("When is it over in... {}?".format(label))
    plt.yscale('log')
    plt.legend()

    if save_to:
        plt.savefig(save_to)
