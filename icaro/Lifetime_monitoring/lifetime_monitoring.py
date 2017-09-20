# A script to compute alpha lifetime.
import os
import sys
import argparse
import datetime

import numpy             as np
import matplotlib.pyplot as plt

import invisible_cities.core.core_functions as coref
import invisible_cities.reco. dst_functions as dstf
import invisible_cities.core. fit_functions as fitf
import invisible_cities.database.load_db    as db

from invisible_cities.icaro.hst_functions import plot
from invisible_cities.icaro.hst_functions import errorbar
from invisible_cities.icaro.hst_functions import hist
from invisible_cities.icaro.hst_functions import hist2d
from invisible_cities.icaro.hst_functions import display_matrix
from invisible_cities.icaro.hst_functions import labels
from invisible_cities.icaro.hst_functions import plot_writer
from invisible_cities.icaro.hst_functions import measurement_string

from icaro.core.kdst_functions import event_rate
from icaro.core.kdst_functions import profile_and_fit
from icaro.core.kdst_functions import time_from_timestamp
from icaro.core.kdst_functions import to_deltatime
from icaro.core.kdst_functions import lifetime_vs_t
from icaro.core.kdst_functions import save_lifetime
from icaro.core.kdst_functions import center_and_fit


np.warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"         ] = 16, 12
plt.rcParams[  "font.size"            ] = 18
plt.rcParams["figure.max_open_warning"] = 1000


space = "\n.\n.\n.\n"
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end=space)


## Configuration
program, *args = sys.argv
parser = argparse.ArgumentParser(program)
parser.add_argument("-r", metavar="run_numbers", type=int, help="run numbers"  ,
                    nargs='+')
parser.add_argument("-t", metavar="run_tags"   , type=str, help="run tags"     ,
                    nargs='+', default=['notag'])
parser.add_argument("-i", metavar="input path" , type=str, help="input path"   )
parser.add_argument("-o", metavar="database"   , type=str, help="database file",
                    default="$ICARODIR/Lifetime_monitoring/Litetimes.txt")
parser.add_argument("-c", metavar="comment"    , type=str, help="comments"     ,
                    default="")
parser.add_argument("-p", metavar="plotsfolder", type=str, help="plots folder" )
parser.add_argument("--save-plots", action="store_true"  ,
                    help="store control plots" )
parser.add_argument("--overwrite" , action="store_true"  ,
                    help="overwrite datebase values" )

flags, extras = parser.parse_known_args(args)

flags.i = os.path.expandvars(flags.i if flags.i else
                             "$DATADIR/")
flags.o = os.path.expandvars(flags.o if flags.o else
                             "$ICARODIR/icaro/Lifetime/Lifetimes.txt")
flags.p = os.path.expandvars(flags.p if flags.p else
                             "$ICARODIR/icaro/Lifetime/Plots/")

run_numbers   = flags.r
run_tags      = flags.t if len(flags.t) == len(flags.r) else flags.t * len(flags.r)
data_filename = flags.i + "{0}/dst_{0}.root.h5"
text_filename = flags.o
run_comment   = flags.c
plots_folder  = flags.p
save_plots    = flags.save_plots
overwrite     = flags.overwrite


Xrange        = -198, 198
Yrange        = -198, 198
Zrange        =    0, 600
Zrange_LT     =   50, 500
Xnbins        =   50
Ynbins        =   50
Znbins        =   50
S1nbins       =   50
S2nbins       =   50

XYrange       = Xrange, Yrange
XYnbins       = Xnbins, Ynbins

Rfiducial     = 100 # in mm
Zcathode      = 500 # in µs

for run_number, run_tag in zip(run_numbers, run_tags):
    run_tag = run_tag.lower()
    savefig = plot_writer(plots_folder + str(run_number), "png")

    if   run_tag == "alphas":
        S1range =    0, 3e3
        S2range =    0, 4e4
    elif run_tag == "kr":
        S1range =    0, 5e1
        S2range =    0, 2e4

    full       = dstf.load_dst(data_filename.format(run_number), "DST", "Events")
    t_begin    = np.min(full.time)
    t_end      = np.max(full.time)
    run_dt     = t_end - t_begin
    full.time -= t_begin
    fid        = full[full.R < Rfiducial] # michel sorel cuts
    cath       = full[full.Z > Zcathode ]
    bulk       = full[full.Z < Zcathode ]

    print(f"# alphas              :"        , len(full))
    print(f"# alphas at R < {Rfiducial} mm:", len(fid ))
    print(f"# alphas at Z > {Zcathode} µs:" , len(cath))
    print(f"# alphas at Z < {Zcathode} µs:" , len(bulk))

    #--------------------------------------------------------
    _, bins, _ = \
    hist(full.time/60, Tnbins)
    labels("Time (min)",
           "Rate (({:.1f} min)$^{{-1}}$)".format(np.diff(bins)[0]),
           "Trigger rate")
    savefig("TriggerRate")

    rate = event_rate(full)
    print("Average trigger rate: {:.2f} Hz".format(rate))

    #--------------------------------------------------------
    plt.figure ()
    plt.subplot(2, 3, 1)
    plt.hist   (full.S1e, S1nbins, S1range)
    labels     ("S1 energy (pes)", "Entries", "S1 energy")

    plt.subplot(2, 3, 2)
    plt.hist   (full.S2e, S2nbins, S2range)
    labels     ("S2 energy (pes)", "Entries", "S2 energy")

    plt.subplot(2, 3, 3)
    plt.hist   (fid .S2e, S2nbins, S2range)
    labels     ("S2 energy (pes)", "Entries", "S2 energy (R < 100 mm)")

    plt.subplot(2, 3, 4)
    plt.hist   (full.Z  ,  Znbins, Zrange)
    labels     ("Drift time (µs)", "Entries", "Drift time")

    plt.subplot(2, 3, 5)
    plt.hist   (fid .Z  ,  Znbins, (Zcathode, Zrange[1]))
    labels     ("Drift time (µs)", "Entries", "Drift time (cathode)")

    plt.subplot(2, 3, 6)
    plt.hist   (fid .Z  ,  Znbins, (Zrange[0], Zcathode))
    labels     ("Drift time (µs)", "Entries", "Drift time (bulk)")

    plt.tight_layout()
    savefig("EZ_distributions")


    #--------------------------------------------------------
    hist2d (full.time, full.S2e, (Tnbins, S2nbins), ((0, full.time.max()), S2range))
    labels ("Time (s)", "Energy (pes)", "Energy vs time")
    savefig("EvsT")


    #--------------------------------------------------------
    plt.figure()

    plt.subplot(2, 3, 1)
    plt.hist2d (full.Z  , full.S2e, ( Znbins, S2nbins), range=( Zrange, S2range))
    labels     ("Drift time (µs)", "S2 energy (pes)", "S2 vs Z")

    plt.subplot(2, 3, 2)
    plt.hist2d (full.Z  , full.S1e, ( Znbins, S1nbins), range=( Zrange, S1range))
    labels     ("Drift time (µs)", "S1 energy (pes)", "S1 vs Z")

    plt.subplot(2, 3, 3)
    plt.hist2d (full.S2e, full.S1e, (S2nbins, S1nbins), range=(S2range, S1range))
    labels     ("S2 energy (pes)", "S1 energy (pes)", "S1 vs S2")

    plt.subplot(2, 3, 4)
    plt.hist2d (fid .Z  , fid .S2e, ( Znbins, S2nbins), range=( Zrange, S2range))
    labels     ("Drift time (µs)", "S2 energy (pes)", "S2 vs Z (R < 100 mm)")

    plt.subplot(2, 3, 5)
    plt.hist2d (fid .Z  , fid .S1e, ( Znbins, S1nbins), range=( Zrange, S1range))
    labels     ("Drift time (µs)", "S1 energy (pes)", "S1 vs Z (R < 100 mm)")

    plt.subplot(2, 3, 6)
    plt.hist2d (fid .S2e, fid .S1e, (S2nbins, S1nbins), range=(S2range, S1range))
    labels     ("S2 energy (pes)", "S1 energy (pes)", "S1 vs S2 (R < 100 mm)")

    plt.tight_layout()
    savefig("S12")


    #--------------------------------------------------------
    plt.figure ()
    plt.subplot(2, 2, 1)
    plt.hist2d (full.X, full.Y, XYnbins, XYrange)
    plt.colorbar().set_label("# events")
    labels     ("X (mm)", "Y (mm)", "XY distribution")

    plt.subplot(2, 2, 2)
    plt.hist2d (fid .X, fid .Y, XYnbins, XYrange)
    plt.colorbar().set_label("# events")
    labels     ("X (mm)", "Y (mm)", "XY distribution (R < 100 mm)")

    plt.subplot(2, 2, 3)
    plt.hist2d (bulk.X, bulk.Y, XYnbins, XYrange)
    plt.colorbar().set_label("# events")
    labels     ("X (mm)", "Y (mm)", "XY distribution (bulk)")

    plt.subplot(2, 2, 4)
    plt.hist2d (cath.X, cath.Y, XYnbins, XYrange)
    plt.colorbar().set_label("# events")
    labels     ("X (mm)", "Y (mm)", "XY distribution (cathode)")

    plt.tight_layout()
    savefig("XY")


    #--------------------------------------------------------
    seed   = S2range[1], -1e3

    E_sel = np.ones_like(fid.Z.values, dtype=bool)
    if run_tag == "kr":
        x, y, y_std = fitf.profileX(fid.Z, fid.S2e, Znbins, Zrange_LT, S2range,
                                    std=True)
        f_low       = fitf.fit(fitf.expo, x, y - y_std, (np.max(y), -1e3)).fn
        f_upp       = fitf.fit(fitf.expo, x, y + y_std, (np.max(y), -1e3)).fn
        E_sel       = coref.in_range(fid.S2e, f_low(fid.Z), f_upp(fid.Z))

    plt.figure()
    F, *_ = profile_and_fit(fid.Z[E_sel], fid.S2e[E_sel],
                            xrange =  Zrange_LT,
                            yrange = S2range,
                            nbins  =  Znbins,
                            fitpar =  seed)
    labels ("Drift time (µs)", "S2 energy (pes)")
    savefig("LifetimeFit")

    E0, u_E0 =  F.values[0], F.errors[0]
    LT, u_LT = -F.values[1], F.errors[1]

    print("Energy at Z=0: ({}) pes".format(measurement_string(E0, u_E0)))
    print("Lifetime     : ({}) µs ".format(measurement_string(LT, u_LT)))
    print("Chi2 fit     : {:.2f}  ".format(F.chi2))

    Zcorr = corrf.LifetimeCorrection(LT, u_LT)

    #--------------------------------------------------------
    plt.figure()
    dst        = fid[E_sel & coref.in_range(fid.Z, *Zrange_LT)]
    timestamps = list(map(time_from_timestamp, dst.time))

    Ts, u_Ts, LTs, u_LTs, E0s, u_E0s = lifetime_vs_t(dst, nslices=8,
                                                     timestamps=timestamps)

    errorbar(Ts, LTs, u_LTs, u_Ts)
    labels  ("Time (s)", "Lifetime (µs)", "Lifetime evolution within run")

    errorbar(Ts, E0s, u_E0s, u_Ts)
    labels  ("Time (s)", "Energy at Z=0 (pes)", "Energy scale evolution within run")

    savefig ("EnergyScaleT")


    #--------------------------------------------------------
    x, y, E, u_E = fitf.profileXY(full.X.values, full.Y.values,
                                  full.S2e.values * Zcorr(full.Z.values).value,
                                  nX, nY, Xrange, Yrange)

    _, cb = display_matrix(x, y, E)
    cb.set_label("E (pes)")
    labels("x (mm)", "y (mm)", "Energy vs XY")
    save("EvsXY")

    corrections = corrf.Correction((x, y), E, u_E, "index", index=(nX//2, nY//2))
    print("Reference energy = {} pes".format(E[tuple(np.argwhere(corrections._fs==1)[0])]))


    #--------------------------------------------------------
    x, y, Q, u_Q = fitf.profileXY(full.X.values, full.Y.values,
                                  full.S2q.values,
                                  nX, nY, Xrange, Yrange)

    _, cb = display_matrix(x, y, Q)
    cb.set_label("Q (pes)")
    labels("x (mm)", "y (mm)", "Charge vs XY")
    save("QvsXY")


    #--------------------------------------------------------
    values, bins = np.histogram(cath.Z, 50)

    mean_val = bins[np.argmax(values)]
    zrange_  = mean_val - 5., mean_val + 5.

    y, x, _  = hist(cath.Z, Znbins, zrange_)
    F        = fitf.fit(fitf.gauss, x, y, (np.max(values), mean_val, 2))

    plot   (x, F.fn(x), lw=3, c='r')
    labels ("Drift length (µs)", "Entries", "Max drift length")
    savefig("MaxDriftLength")

    max_drift   = F.values[1]
    u_max_drift = F.errors[1]
    v_drift     = db.DetectorGeo().ZMAX[0]/max_drift
    u_v_drift   = v_drift*u_max_drift/max_drift

    print("Drift time     : ({}) µs   ".format(measurement_string(max_drift,
                                                                  u_max_drift)))
    print("Drift velocity : ({}) mm/µs".format(measurement_string(  v_drift,
                                                                    u_v_drift)))


    #--------------------------------------------------------
    date_begin = time_from_timestamp(t_begin)
    date_end   = time_from_timestamp(t_end  )
    date_lapse = to_deltatime       (t_begin, t_end, unit="s", to_str=True)

    save_lifetime(text_filename,
                     run_number,    run_tag,
                             LT,       u_LT,
                             E0,       u_E0,
                        v_drift,  u_v_drift,
                        t_begin,      t_end,     run_dt,
                     date_begin,   date_end, date_lapse,
                     comment   = run_comment.replace(" ", "_"),
                     delimiter = " ",
                     overwrite = overwrite)
    print("", end=space)
