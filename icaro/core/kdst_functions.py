import os
import datetime

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import invisible_cities.core.fit_functions  as  fitf
import invisible_cities.core.core_functions as coref
import invisible_cities.reco.dst_functions  as dstf

from invisible_cities.evm  .ic_containers import Measurement
from invisible_cities.icaro.hst_functions import plot
from invisible_cities.icaro.hst_functions import errorbar
from invisible_cities.icaro.hst_functions import labels


xy_map = dstf.load_xy_corrections("../Na_June_Axial/merged_corrections.h5")



def create_file_if_neccessary(filename):
    if not os.path.exists(filename):
        open(filename, "w").close()


def delete_lifetime_entry(filename, run_number, delimiter=" ", overwrite=False):
    in_data          = open(filename, "r").readlines()
    if not in_data:
        return True

    header, *in_data = in_data
    out_data         = list(filter(lambda line: int(line.split(delimiter)[0]) != run_number, in_data))

    if len(in_data) == len(out_data): return True
    if overwrite or input("Overwrite value for run {} (y/n)? ".format(run_number)) == "y":
        open(filename, "w").write(header + "".join(out_data))
        return True
    return False


def save_lifetime(    filename,
                    run_number,    run_tag,
                            lt,       u_lt,
                           E_0,       u_E0,
                       v_drift,  u_v_drift,
                       t_start,      t_end,            dt,
                    date_start,   date_end,         ddate,
                    comment   = "" ,
                    delimiter = " ",
                    overwrite = False):
    create_file_if_neccessary(filename)
    if not delete_lifetime_entry(filename, run_number, overwrite=overwrite):
        return

    line = delimiter.join(map(str, [run_number,    run_tag,
                                            lt,       u_lt,
                                           E_0,       u_E0,
                                       v_drift,  u_v_drift,
                                      t_start ,      t_end,    dt,
                                    date_start,   date_end, ddate,
                                       comment]))

    in_data          = open(filename, "r").readlines()
    in_data.append(line + "\n")

    header, *in_data = in_data
    in_data          = filter(lambda x: not x.isspace(), in_data)
    out_data         = sorted(in_data, key=lambda x: int(x.split(delimiter)[0]))
    open(filename, "w").write(header + "".join(out_data))


def load_lifetimes(filename, delimiter=" ", **kwargs):
    return pd.read_csv(filename, sep=delimiter, **kwargs)


def datetime_to_str(datetime, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.strftime(tformat)


def time_from_timestamp(timestamp, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.datetime.fromtimestamp(timestamp).strftime(tformat)


def str_to_datetime(timestamp, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.datetime.strptime(timestamp, tformat)


def to_deltatime(t0, t1, unit="s", to_str=False):
    delta = pd.Timedelta(t1 - t0, unit=unit)
    return str(delta).replace(" ", "-") if to_str else delta


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

    CHI2    = []
    LAMBDA  = []
    ELAMBDA = []
    TSTAMP  = []

    for i in range(nbins):
        k0 =  i    * indx
        k1 = (i+1) * indx - 1

        print(' ---fit over events between {} and {}'.format(k0, k1))
        st0 = time_from_timestamp(dst.time.values[k0])
        st =  time_from_timestamp(dst.time.values[k1])

        print('time0 = {} time1 = {}'.format(st0, st))

        tleg = dst[coref.in_range(dst.time.values,
                                  minval = dst.time.values[k0],
                                  maxval = dst.time.values[k1])]

        print('size of time leg = {}'.format(len(tleg)))
        F, x, y, u_x, u_y = profile_and_fit(tleg.Z, tleg.S2e,
                                            xrange = zrange,
                                            yrange = Erange,
                                            nbins  = nbins,
                                            fitpar = (50000, -300))
        labels("Drift time (µs)", "S2 energy (pes)", "Lifetime fit")
        print_fit(F)

        chi = chi2(F, x, y, u_y)
        print('chi2 = {}'.format(chi))

        CHI2   .append(chi)
        LAMBDA .append(F.values[1])
        ELAMBDA.append(F.errors[1])
        TSTAMP .append(st)

    TIME = [datetime.datetime.strptime(elem,
           '%Y-%m-%d %H:%M:%S') for elem in TSTAMP]

    return CHI2, LAMBDA, ELAMBDA, TSTAMP, TIME


def lifetime_vs_t(dst, nslices=10, nbins=10, seed=(3e4, -5e2), fit_range=[100,500],
                  timestamps=None, sliced=False, slparam={} ,**profOpt):
    LT, LTu = [], []
    E0, E0u = [], []
    T ,  Tu = [], []

    slparam_base={'Zrange':[0,550], 'Yrange':[4e3, 16e3], 'nZbins':50, 'step':100,
                  'nbin_fit':100, 'seed_fit':[1e4, 12e3, 200], 'range_fit':[11e3,16e3]}

    tmin = np.min(dst.time)
    tmax = np.max(dst.time)
    bins = np.linspace(tmin, tmax, nslices+1)
    for t0, t1 in zip(bins[:-1], bins[1:]):
        subdst   = dst[coref.in_range(dst.time, t0, t1)]
        if sliced:
            Z, E, Eu = get_E_mean_vs_z(subdst,**{**slparam_base, **slparam})

        else:
            Z, E, Eu = fitf.profileX(subdst.Z, subdst.S2e, nbins, **profOpt)

        sel_Z    = fitf.in_range(Z, *fit_range)
        f        = fitf.fit(fitf.expo, Z[sel_Z], E[sel_Z], seed, sigma=Eu[sel_Z])



        LT .append(-f.values[1] )
        LTu.append( f.errors[1] )
        E0 .append( f.values[0] )
        E0u.append( f.errors[0] )
        T  .append(0.5*(t1 + t0))
        Tu .append(0.5*(t1 - t0))

    return T, Tu, LT, LTu, E0, E0u


def event_rate(kdst):
    t0 = np.min(kdst.time)
    t1 = np.max(kdst.time)
    return kdst.event.size/(t1-t0)


def profile_and_fit(X, Y, xrange, yrange, nbins, fitpar, fitOpt  = "r"):
#    n_it = 0
#    chi2 = 0
#    while not 0.8 < chi2 < 1.4:
    u_x       = 0.5*(xrange[1] - xrange[0])/nbins
    x, y, u_y = fitf .profileX(X, Y, nbins=nbins, xrange=xrange, yrange=yrange)
    f         = fitf.fit(fitf.expo, x, y, fitpar, sigma=u_y)
#        chi2      = f.chi2
#        nbins    -= n_it

#        n_it += 1
#        if nbins < 5:
#            print("Chi2 does not get close to 1")
#            break

    errorbar(x, y, u_y, u_x,
             linestyle = "none",
             marker    = ".")
    plot    (x, f.fn(x),
             fitOpt,
             new_figure=False)
    return f, x, y, u_x, u_y



def print_fit(f):
    for i, val in enumerate(map(Measurement, f.values, f.errors)):
        print("Parameter {}: {}".format(i, val))



def get_E_mean_vs_z(data, Zrange=[0,550], Yrange=[4e3, 16e3], nZbins=50, step=100,
                    nbin_fit=100, seed_fit=[1e4, 12e3, 200], range_fit=[11e3,16e3]):

    n_seed, mean_seed, sigma_seed = seed_fit
    min_fin, max_fit              = range_fit

    Zbins              = np.linspace(*Zrange, nZbins)
    Z_drift, u_Z_drift = np.zeros_like(Zbins[:-1]), np.zeros_like(Zbins[:-1])

    for i in range(len(Zbins)-1):

        data_   = data[fitf.in_range(data.Z.values, Zbins[i], Zbins[i+1])]
        E_zcorr = data_.S2e.values * xy_map(data_.X.values, data_.Y.values).value
        y, x    = np.histogram(E_zcorr, nbin_fit, range=Yrange)
        x       = x[1:]-np.diff(x)*0.5

        mean_i    = mean_seed-i*step
        min_fin_i = min_fin-i*step

        seed =     n_seed,    mean_i,     sigma_seed
        down = 0.1*n_seed, min_fin_i, 0.8*sigma_seed
        up   = 1e4*n_seed,   max_fit,  10*sigma_seed

        f    = fitf.fit(fitf.gauss, x, y, seed,
                        fit_range=(min_fin-i*step, max_fit),
                        bounds = [down,up])

        Z_drift[i], u_Z_drift[i] = f.values[1],f.errors[1]

    return Zbins[1:]-np.diff(Zbins)*0.5, Z_drift, u_Z_drift

def center_and_fit(data, range_fit, nbins, fitpar, fitOpt  = "r", xrange=[0,550], yrange=[4e3, 16e3],
                    step_slice=75, nbin_slice=100, seed_slice=[1e4, 12e3, 200],
                    range_slice=[11e3,16e3], plot_lims=True):

    u_x       = 0.5*(xrange[1] - xrange[0])/nbins
    x, y, u_y = get_E_mean_vs_z(data, Zrange=xrange, Yrange=yrange,
                                nZbins=nbins, step=step_slice,
                                nbin_fit=nbin_slice, seed_fit= seed_slice,
                                range_fit=range_slice)
    sel_range = fitf.in_range(x, *range_fit)
    f         = fitf.fit(fitf.expo, x[sel_range], y[sel_range], fitpar,
                         sigma=u_y[sel_range])
    plt.figure()
    plt.errorbar(x, y, u_y, u_x, linestyle="none", marker=".")
    plt.plot    (x, f.fn(x), fitOpt)

    if plot_lims:
        plt.axvline (range_fit[0], c='k', ls='dashed')
        plt.axvline (range_fit[1], c='k', ls='dashed')
    return f, x, y, u_x, u_y
