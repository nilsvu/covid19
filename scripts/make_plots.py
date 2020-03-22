import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import locale
import matplotlib.dates as mdates
import logging
import yaml
from scripts.load_data import load_jhu_data
from scripts.plot import *


if __name__ == "__main__":
    try:
        locale.setlocale(locale.LC_TIME, "de_DE")
    except locale.Error:
        logging.warning("Unable to set locale")

    import os
    os.makedirs('plots', exist_ok=True)
    plot_timeshifts('plots/germany_total.svg')
    plot_timeshifts('plots/germany_total_log.svg', log=True)
    plot_german_states('plots/states.svg')
    plot_german_states('plots/states_log.svg', log=True)

    data_jhu = load_jhu_data()
    for country in yaml.safe_load(open('data/fits.yaml')):
        plot_prediction(
            data_jhu[country['Dataset']][country['FitOnset']:],
            label=country['Name'],
            save_to='plots/prediction_{}.svg'.format(country['Slug']))
