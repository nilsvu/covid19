import datetime
import numpy as np
import scipy.optimize as opt


def logistic(t, a, b, c):
    return c / (1 + np.exp(-(t - b) / a))


def logistic_deriv(t, a, b, c):
    return np.exp(-(t - b) / a) * c / (a * (1 + np.exp(-(t - b) / a))**2)


# The date when the logistic function has reached a fraction `perc_flat` of
# its asymptote
def get_logistic_date_flat(params, perc_flat=0.98):
    t_flat = opt.fsolve(
        lambda x: logistic(x, *params) - perc_flat * int(params[2]), params[1])
    return t_flat, datetime.date.today() + datetime.timedelta(days=int(t_flat))


# The date when the derivative of the logistic function has declined below
# `below_num_new_cases`
def get_logistic_date_end(params, below_num_new_cases=10):
    t_end = opt.fsolve(
        lambda x: logistic_deriv(x, *params) - below_num_new_cases,
        params[1] + 10)
    return t_end, datetime.date.today() + datetime.timedelta(days=int(t_end))


def fit_model(data, model=logistic, p0=None, *args, **kwargs):
    if p0 is None:
        p0 = [4, 0, np.max(data.values)]
    t_days = np.array([(t.to_pydatetime().date() - datetime.date.today()).days
                       for t in data.index])
    popt, pcov = opt.curve_fit(model,
                               t_days,
                               data.values,
                               sigma=np.sqrt(data.values),
                               p0=p0,
                               *args,
                               **kwargs)
    sigma = np.sqrt(np.diag(pcov))
    return popt, sigma, t_days[0]
