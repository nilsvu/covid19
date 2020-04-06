import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import datetime
import yaml
import numpy as np
import pandas as pd
import pycountry
import gettext
from .analyze import fit_model, get_logistic_date_end, logistic, logistic_deriv
from .load_data import load_jhu_data, load_cds_data


def plot_daily_new_cases(save_to=None, average_over_days=7, min_total_cases=5e2):
    data = load_cds_data()['cases']
    german = gettext.translation('iso3166',
                                 pycountry.LOCALES_DIR,
                                 languages=['de'])
    plt.clf()
    plt.figure(figsize=(10, 7.5))
    averaged_data_germany = None
    alpha_scale = np.log(data.loc['Germany'].iloc[-1]) - np.log(min_total_cases)
    for country, data_country in data.groupby(level='country'):
        if data_country.iloc[-1] < min_total_cases:
            continue
        # Switzerland has very inconsistent data: https://github.com/covidatlas/coronadatascraper/issues/460
        if country in ['Switzerland']:
            continue
        try:
            found_country_data = pycountry.countries.search_fuzzy(country)
            label = german.gettext(found_country_data[0].name)
        except LookupError:
            label = country
        averaged_data = np.convolve(data_country, np.ones(average_over_days) / average_over_days, mode='valid')
        if country == 'Germany':
            style = dict(color='black', lw=2, alpha=1, zorder=100)
            annotation_style = dict(bbox=dict(fc=(1, 1, 1, 0.8), ec=(0, 0, 0, 0.2), pad=2))
            averaged_data_germany = averaged_data
        else:
            style = dict(color=None, lw=1,
                alpha=0.8 * max(0, min(1, (np.log(averaged_data[-1]) - np.log(min_total_cases)) / alpha_scale)),
                zorder=10)
            annotation_style = {}
        averaged_data_diff = np.diff(averaged_data)
        plt.plot(averaged_data[1:], averaged_data_diff, label=label, **style)
        plt.scatter(averaged_data[-1], averaged_data_diff[-1], color=style['color'], zorder=style['zorder'],
            alpha=style['alpha'])
        plt.annotate(label, xy=(averaged_data[-1], max(averaged_data_diff[-1], 1.1e2)),
            textcoords='offset points', xytext=(8, 0), va='center', zorder=style['zorder'], **annotation_style,
            alpha=style['alpha'])
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Tägliche Neuansteckungen (Wochenmittelwert)")
    plt.xlabel("Ansteckungen Gesamt")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    germany_today = averaged_data_germany[-1] - averaged_data_germany[-2]
    plt.axhline(germany_today, ls='dashed', color='lightgray', zorder=1)
    plt.annotate("${:.0f}$ Neuansteckungen pro Tag im Mittel über {} Tage".format(germany_today, average_over_days), xy=(5e2, germany_today), textcoords='offset points', xytext=(8, 4), va='bottom', zorder=100)
    plt.annotate("(${:.0f}$ Neuansteckungen am {:%-d. %B %Y})".format((data.loc['Germany'].values[-1] - data.loc['Germany'].values[-2]), data.loc['Germany'].index[-1]), xy=(5e2, germany_today), textcoords='offset points', xytext=(8, -4), va='top', zorder=100)
    plt.xlim(left=min_total_cases)
    plt.ylim(bottom=1e2)
    if save_to:
        plt.savefig(save_to)


def plot_timeshifts(save_to=None, log=False):
    data = load_cds_data()['cases']

    plt.clf()
    plt.figure(figsize=(10, 6))

    max_timeshift = datetime.timedelta(days=0)
    for timeshift_entry in yaml.safe_load(open('data/timeshifts.yaml', 'r')):
        data_shifted = data.loc[timeshift_entry['Dataset']].copy()
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

    data.loc['Germany'].plot(marker='.',
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
    data = load_cds_data('state', ['Germany'])['cases']
    translations = yaml.safe_load(open('data/translation_states.yaml', 'r'))

    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.title("Ansteckungen pro Bundesland in Deutschland")
    for state, data_state in data.groupby('state'):
        data_state.index = data_state.index.droplevel(0)
        data_state.plot(marker='.', ls='-')
        plt.annotate(translations[state], (data_state.index[-1], data_state.iloc[-1]),
                     textcoords='offset points',
                     xytext=(8, 0),
                     bbox=dict(fc=(1, 1, 1, 0.5), ec=(1, 1, 1, 0), pad=2))
    plt.xlabel(None)
    plt.ylabel("Ansteckungen Gesamt")
    plt.xlim(right=datetime.date.today() + datetime.timedelta(days=3))
    plt.axvline(datetime.date.today(), alpha=0.2, color='black', lw=3)

    plt.grid(color=(0.9, 0.9, 0.9), lw=1, axis='y')

    if log:
        plt.yscale('log')

    if save_to:
        plt.savefig(save_to)


def plot_prediction(data, label, *args, save_to=None, **kwargs):
    for i, d in enumerate(data):
        if d > 50:
            data = data[i:]
            break
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
