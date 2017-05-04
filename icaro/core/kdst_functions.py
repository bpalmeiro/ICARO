import numpy as np
import tables as tb
import pandas as pd
import invisible_cities.core.fit_functions as fitf
from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.core.mpl_functions import set_plot_labels
from invisible_cities.core.core_functions import in_range
import matplotlib.pyplot as plt
from   collections import namedtuple
import datetime



def time_from_timestamp(timestamp, tformat='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(timestamp).strftime(tformat)


def lifetime(dst, zrange=(25,530), Erange=(1e+3, 70e3), nbins=10):
    """Compute lifetime as a function of t."""

    print('using data set with length {}'.format(len(dst)))
    st0 = time_from_timestamp(dst.time.values[0])
    st1 = time_from_timestamp(dst.time.values[-1])
    it0 = 0
    it1 = len(dst)
    print('t0 = {} (index = {}) t1 = {} (index = {})'.format(st0, it0, st1, it1))

    indx  = int(len(dst) / nbins)
    print('bin length = {}'.format(indx))

    CHI2 = []
    LAMBDA = []
    ELAMBDA = []
    TSTAMP = []

    for i in range(nbins):
        k0 = i * indx
        k = (i+1) * indx - 1
        print(' ---fit over events between {} and {}'.format(k0, k))
        st0 = time_from_timestamp(dst.time.values[k0])
        st =  time_from_timestamp(dst.time.values[k])

        print('time0 = {} time1 = {}'.format(st0,st))

        tleg = dst[in_range(dst.time.values, minval=dst.time.values[k0], maxval=dst.time.values[k])]
        print('size of time leg = {}'.format(len(tleg)))
        F, x, y, sy = profile_and_fit(tleg.Z, tleg.S2e,
                                      xrange=zrange,
                                      yrange=Erange,
                                      nbins=nbins,
                                      fitpar=(50000,-300),
                                      label=("Drift time ($\mu$s)", "S2 energy (pes)"))
        print_fit(F)
        chi = chi2(F, x, y, sy)
        print('chi2 = {}'.format(chi))
        CHI2.append(chi)
        LAMBDA.append(F.values[1])
        ELAMBDA.append(F.errors[1])
        TSTAMP.append(st)

    TIME = [datetime.datetime.strptime(elem,
           '%Y-%m-%d %H:%M:%S') for elem in TSTAMP]

    return CHI2, LAMBDA, ELAMBDA, TSTAMP, TIME


class MapXY:
    def __init__(self, x, y, E):
        self.xs = x.reshape(x.size, 1) #file to column vector
        self.ys = y.reshape(y.size, 1)
        self.eref = E[E.shape[0]//2, E.shape[1]//2]
        self.es = E
        print('reference energy = {}'.format(self.eref))

    def xycorr(self, x, y):
        x_closest = np.apply_along_axis(np.argmin, 0, abs(x - self.xs))
        y_closest = np.apply_along_axis(np.argmin, 0, abs(y - self.ys))
        e = self.es[x_closest, y_closest]
        e[ e < 1e3] = self.eref
        return self.eref / e


def load_dst(filename):
    with tb.open_file(filename) as h5:
        return pd.DataFrame.from_records(h5.root.DST.Events.read())


def event_rate(kdst):
    t0 = np.min(kdst.time)
    t1 = np.max(kdst.time)
    return kdst.event.size/(t1-t0)


def profile_and_fit(X, Y, xrange, yrange, nbins, fitpar, label):
    fitOpt  = "r"
    xe = (xrange[1] - xrange[0])/nbins

    x, y, sy = fitf.profileX(X, Y, nbins=nbins,
                             xrange=xrange, yrange=yrange, drop_nan=True)
    sel  = fitf.in_range(x, xrange[0], xrange[1])
    x, y, sy = x[sel], y[sel], sy[sel]
    f = fitf.fit(fitf.expo, x, y, fitpar, sigma=sy)

    plt.errorbar(x=x, xerr=xe, y=y, yerr=sy,
                 linestyle='none', marker='.')
    plt.plot(x, f.fn(x), fitOpt)
    set_plot_labels(xlabel=label[0], ylabel=label[1], grid=True)
    return f, x, y, sy


def profile_and_fit_radial(X, Y, xrange, yrange, nbins, fitpar, label):
    fitOpt  = "r"
    xe = (xrange[1] - xrange[0])/nbins

    x, y, sy = fitf.profileX(X, Y, nbins=nbins,
                             xrange=xrange, yrange=yrange, drop_nan=True)
    sel  = fitf.in_range(x, xrange[0], xrange[1])
    x, y, sy = x[sel], y[sel], sy[sel]
    f = fitf.fit(fitf.polynom, x, y, fitpar, sigma=sy)

    plt.errorbar(x=x, xerr=xe, y=y, yerr=sy,
                 linestyle='none', marker='.')
    plt.plot(x, f.fn(x), fitOpt)
    set_plot_labels(xlabel=label[0], ylabel=label[1], grid=True)
    return f, x, y, sy


def chi2(F, X, Y, SY):
    fitx = F.fn(X)
    n = len(F.values)
    print('degrees of freedom = {}'.format(n))
    chi2t = 0
    for i, x in enumerate(X):
        chi2 = abs(Y[i] - fitx[i])/SY[i]
        chi2t += chi2
        #print('x = {} f(x) = {} y = {} ey = {} chi2 = {}'.format(
               #x, fitx[i], Y[i], SY[i], chi2 ))
    return chi2t/(len(X)-n)

    #chi2 = np.sum(np.ma.masked_invalid((fitx - y)**2/sy**2))
    #print('chi2 = {}'.format(chi2))

def print_fit(f):
    for i, val in enumerate(f.values):
        print('fit par[{}] = {} error = {}'.format(i, val, f.errors[i]))
