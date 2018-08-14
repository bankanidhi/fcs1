import glob
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from lmfit import Model
matplotlib.style.use("seaborn-colorblind")


def return_corr_function_data(filename):
    """This function opens ascii files and reads the correlation function raw
    data
    """
    b = np.genfromtxt(
        filename, delimiter="\t", usecols=(0), skip_header=50,
        max_rows=186).T

    b = np.column_stack((b, np.genfromtxt(
        filename, delimiter="\t", usecols=(3), skip_header=50,
        max_rows=186) - 1))
    # Returning transposed matrix for easy iteration
    return b.T


def fit_model(t, y):
    gmodel = Model(correlation)
    gmodel.set_param_hint('g0', value=0.1)
    gmodel.set_param_hint('tauD', value=1e-4, min=1e-6, max=100)
    gmodel.set_param_hint('sp', value=0.01, min=0.001, max=0.1)
    gmodel.set_param_hint('bl', value=1e-6)
    pars = gmodel.make_params()
    return gmodel.fit(y, pars, t=t)


def correlation(t, g0, tauD, sp, bl):
    return g0 / ((1 + t / tauD) * (1 + sp * t / tauD)**(0.5)) + bl


# def generate_report(result):
#     print(result.fit_report())


def analyse_data_single(filename, lowlimit=1e-5, highlimit=1):
    corr_data = return_corr_function_data(filename=filename)

    t = corr_data[0]

    mask = (t > lowlimit)  # * (t < highlimit)

    # Limit the time axis in the raw data
    useful_t = corr_data[0][mask]

    # Apply the same mask on all the y data
    useful_y_list = [y[mask] for y in corr_data[1:]]
    fit_param = fit_model(useful_t, useful_y_list)
    plot_fits(corr_data, fit_param=fit_param, mask=mask)
    return fit_param


# def plot_fits(corr_data, fit_param, mask):
#     plt.plot(corr_data[0], corr_data[1], 'o', label="Raw data")
#     plt.plot(corr_data[0][mask], fit_param.best_fit, '-r',
#              label="Fit parameters")
#     plt.plot(corr_data[0][mask], fit_param.residual, '-g', label="Residuals")
#     plt.xscale('log')
#     plt.xlabel('Delay time (s)', fontsize=14)
#     plt.ylabel('Autocorrelation', fontsize=14)
#     plt.text(0.5, 0.5,
#              "tauD = " +
#              '%.1f' % (fit_param.best_values["tauD"] * 1e6) + ' $\mu s$' + '\n'
#              + "g0 = " + '%.2E' % (fit_param.best_values["g0"]) + '\n'
#              "sp = " + '%.2E' % (fit_param.best_values["sp"]), horizontalalignment='center', verticalalignment='center', transform=plt.plot.transAxes)
#     plt.legend()
#     plt.show()


def plot_fits(corr_data, fit_param, mask):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(corr_data[0], corr_data[1], 'o', label="Raw data")
    ax.plot(corr_data[0][mask], fit_param.best_fit, '-r',
            label="Fit parameters")
    ax.plot(corr_data[0][mask], fit_param.residual, '-g', label="Residuals")
    ax.set_xscale('log')
    ax.set_xlabel('Delay time (s)', fontsize=14)
    ax.set_ylabel('Autocorrelation', fontsize=14)
    ax.text(0.7, 0.6,
            "tauD = " +
            '%.1f' % (fit_param.best_values["tauD"] * 1e6) + ' $\mu s$' + '\n'
            + "g0 = " + '%.2E' % (fit_param.best_values["g0"]) + '\n'
            "sp = " + '%.2E' % (fit_param.best_values["sp"]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)
    ax.legend()
    plt.show()


def find_files(keyword="./*.sin"):
    return sorted(glob.glob(keyword))
