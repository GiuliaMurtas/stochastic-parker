import argparse
import errno
import math
import json
import os

import h5py
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from scipy.constants import physical_constants
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import gridspec

from scipy.stats import linregress

import sys
sys.path.insert(0, '/anvil/scratch/x-gmurtas/athena/vis/python')
import athena_read
from sde_util import *

plt.rcParams['figure.dpi'] = 400 # this defines the resolution of the window/plot

tableau_colors = [
    'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]

cwd = os.getcwd() # returns current working directory of a process

################################
## MHD SIMULATION INFORMATION ##
################################

def extend_mhd_info(mhd_run):
    """Extend mhd_info based on the SDE runs
    """
    # normalizations
    L0 = 5.0E9 # 5.0E6 for flares   # in m
    b0 = 0.002 # 50.0  for flares   # in Gauss
    n0 = 1.5e3 # 1.0e10 for flares  # number density in cm^-3
    lscale = 1.0
    bscale = 1.0
    L0 *= lscale
    b0 *= bscale
    n0 *= bscale**2  # to keep the Alfven speed unchanged
    va = calc_va_cgs(n0, b0) / 1E2  # m/s
    t0 = L0 / va  # in sec
    normalization = {"L0": L0, "b0": b0, "n0": n0, "va": va, "t0": t0}
    print(r'va =',va,r'm/s')
    
    # Read the file
    filename = cwd + "/mhd_runs_for_sde.json"
    with open(filename, 'r') as fh: # reads the file mhd_runs_for_sde.json
        configs = json.load(fh)
    mhd_run_dir = configs[mhd_run]["run_dir"] # here the directory where the data are is selected
    config_tmp = load_mhd_config(mhd_run_dir)  # MHD configuration for SDE runs
    keys = config_tmp.dtype.names
    print(r'Keys:', keys)
    mhd_config_for_sde = {key: config_tmp[key][0] for key in keys}
    mhd_config_for_sde.update(configs[mhd_run])
    mhd_info = get_mhd_info(mhd_run_dir, configs[mhd_run]["config_name"])
    mhd_info["norm"] = normalization
    
    # get the time intervals for different outputs
    for i in range(1, 100):
        vname = "output" + str(i)
        if vname in mhd_info:
            dtv = "dt_" + mhd_info[vname]["file_type"]
            mhd_info[dtv] = mhd_info[vname]["dt"] * mhd_info["norm"]["t0"] # normalization of time outputs in seconds
        else:
            break
    mhd_info.update(mhd_config_for_sde)
    return mhd_info
    
mhd_run = "test_1024_2048"
# The command above looks out for the file athinput.reconnection

# Not only: this folder is the same indicated during the run,
# when the 'bin_data' folder is created. Changing the directory
# of the Athena++ data here should involve changing the directory
# of bin_data.

# Also, this should be coordinated with mhd_runs_for_sde.json
mhd_info = extend_mhd_info(mhd_run)

#########################################
## GLOBAL ENERGY SPECTRUM - SINGLE ION ##
#########################################

def get_ebins_kev(sde_run_config, pbins_edges):
    """get energy bins in kev

    Arguments:
        sde_run_config (dict): SDE run configuration
        pbins_edges: momentum bins edges
    """
    if "species" in sde_run_config:
        species = sde_run_config["species"]
    else:
        species = "electron"
    mtext = species + " mass energy equivalent in MeV"
    rest_ene_kev = physical_constants[mtext][0] * 1E3
    e0_kev = sde_run_config["e0"]
    gamma = e0_kev / rest_ene_kev + 1.0
    p0_mc = math.sqrt(gamma**2 - 1)  # p0 in terms of mc
    tmin = sde_run_config["tmin"]
    run_name = sde_run_config["run_name"]
    pinit = 0.1  # commonly used values
    pbins = 0.5 * (pbins_edges[1:] + pbins_edges[:-1])
    pbins_mc = pbins * p0_mc / pinit
    pbins_edges_mc = pbins_edges * p0_mc / pinit
    gamma_bins = np.sqrt(pbins_mc**2 + 1)
    gamma_bins_edges = np.sqrt(pbins_edges_mc**2 + 1)
    ebins_kev = (gamma_bins - 1) * rest_ene_kev
    ebins_edges_kev = (gamma_bins_edges - 1) * rest_ene_kev
    debins_kev = np.diff(ebins_edges_kev)
    
    return (ebins_kev, ebins_edges_kev, debins_kev)

def plot_global_spectra(sde_run_config):
    """Plot global spectra

    Arguments:
        sde_run_config (dict): SDE run configuration
    """
    kwargs = {
        "color": 'r',
        "show_plot": False,
        "plot_power": True,
        "power_test": False,
        "label_text": r"Protons"
    }
    tmin = sde_run_config["tmin"]
    tmax = sde_run_config["tmax"]

    run_name = sde_run_config["run_name"]
    spl = run_name.split("/") # divides "run_name" in spectrum_config.json where it arrives to the symbol "/"
    # printing spl results, i.e., in ['athena_reconnection_test', 'reconnection_test_0', 'transport_test_run_0']
    mhd_run = spl[0] # i.e. athena_reconnection_test
    sde_run = spl[1] # i.e. reconnection_test_0
    # these are corrected two blocks later (where I modified to get to the last nested folder)

    rect = [0.16, 0.16, 0.8, 0.8]
    fig1 = plt.figure(figsize=[3.5, 2.5])
    ax1 = fig1.add_axes(rect)
    
    #for tframe in range(tmin, tmax + 1): # for cycle going through every 10 time outputs
    for tframe in range(tmin, tmax + 1,10): # for cycle going through all the time outputs
        fname = "../../" + run_name + "/fdists_" + str(tframe).zfill(4) + ".h5"
        with h5py.File(fname, "r") as fh:
            pbins_edges = fh["pbins_edges_global"][:]
            fdist = fh["fglobal"][:, :]
        dnptl_bins = np.sum(fdist, axis=1) # number of particles in each bin
        pbins = 0.5 * (pbins_edges[1:] + pbins_edges[:-1])
        if tframe == tmin:
            ebins_kev, _, debins_kev = get_ebins_kev(sde_run_config, pbins_edges)
        fe = dnptl_bins / debins_kev # number of particles in each bin * momentum bin size / energy bin size in keV
        flux = dnptl_bins * pbins / debins_kev
        kwargs["plot_power"] = False
        kwargs["color"] = plt.cm.jet((tframe - tmin) / float(tmax - tmin), 1) #jet as original scale #'red'
        kwargs["label_text"] = r't = '+str(round(0.1*tframe,2))+r' $\tau_A$'
        plot_energy_spectrum(ebins_kev, flux, ax1, sde_run_config, tframe, **kwargs)
        #plt.plot(ebins_kev[42:70], flux[42:70],linestyle='dashed',color='red')
        #plt.plot(ebins_kev[49:62], flux[49:62],linestyle='dashed',color='red')
        #plt.plot(ebins_kev[65:75], flux[65:75],linestyle='dashed',color='cyan')
        
        #slopea, intercepta, r_valuea, p_valuea, std_erra = linregress(np.log10(ebins_kev[42:70]),np.log10(flux[42:70]))
        #print(slopea,std_erra)
        #fe_fakea = np.power((ebins_kev/ebins_kev[73]),slopea) * intercepta
        #plt.plot(ebins_kev[42:70], fe_fakea[42:70],linestyle='dashed',linewidth=1,color='black')
        
        #slopeb, interceptb, r_valueb, p_valueb, std_errb = linregress(np.log10(ebins_kev[65:75]),np.log10(flux[65:75]))
        #print(slopeb,std_errb)
        #fe_fakeb = np.power((ebins_kev/ebins_kev[65]),slopeb) * interceptb
        #plt.plot(ebins_kev[65:75], fe_fakeb[65:75],linestyle='dashed',linewidth=1,color='black')
        
        #for i in range(71,80):
        #    print(r'Flux ratio:',2.72-fe_fakea[i]/flux[i],r'Energy:',ebins_kev[i],i)
        #plt.plot(ebins_kev[76], flux[76], 'rX')

    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.set_xlabel(r'$\varepsilon$ (keV)', fontsize=10)
    ax1.set_ylabel(r'$J$', fontsize=10)
    ax1.tick_params(which='both',labelsize=8,direction='in')
    ax1.grid(True)
    ax1.set_xlim([0.01, 1000])
    ax1.set_ylim([0.001,100000000])
    ax1.legend(loc="lower left",prop={'size': 6}, borderpad=0.5,labelspacing=0.5,ncol =2)
    
    fdir = "../img/global_spectrum/" + mhd_run + "/"
    mkdir_p(fdir)
    fname = fdir + "espect_" + sde_run + ".jpg"
    fig1.savefig(fname, dpi=400)
    plt.show()
    
# Load the configuration for SDE runs
with open('spectrum_config.json', 'r') as file_handler:
    config = json.load(file_handler)
    
# The file 'spectrum_config.json' must be modified before changing sde_run!
sde_run = "transport_He_1"
run_name = "athena_reconnection/" + mhd_run + "/" + sde_run
sde_run_config = config[run_name]

p0 = 0.1
e0 = sde_run_config["e0"] / (0.5*p0**2)
sde_run_config["tmin"] = 30
sde_run_config["tmax"] = 80
plot_global_spectra(sde_run_config)

