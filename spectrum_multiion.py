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

cwd = os.getcwd() # returns current working directory of a process

################################
## MHD SIMULATION INFORMATION ##
################################

def extend_mhd_info(mhd_run):
    """Extend mhd_info based on the SDE runs
    """
    # normalizations
    L0 = 5.0E9  # in m
    b0 = 0.002  # in Gauss
    n0 = 1.5e3  # number density in cm^-3
    lscale = 1.0
    bscale = 1.0
    L0 *= lscale
    b0 *= bscale
    n0 *= bscale**2  # to keep the Alfven speed unchanged
    va = calc_va_cgs(n0, b0) / 1E2  # in m/s
    t0 = L0 / va  # in sec
    normalization = {"L0": L0, "b0": b0, "n0": n0, "va": va, "t0": t0}
    print(r'va =',va,r'm/s')
    
    # Read the file
    filename = cwd + "/mhd_runs_for_sde.json"
    with open(filename, 'r') as fh: 
        configs = json.load(fh)
    mhd_run_dir = configs[mhd_run]["run_dir"] 
    config_tmp = load_mhd_config(mhd_run_dir)  
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
            mhd_info[dtv] = mhd_info[vname]["dt"] * mhd_info["norm"]["t0"] 
        else:
            break
    mhd_info.update(mhd_config_for_sde)
    return mhd_info
    
mhd_run = "test_1024_2048"
mhd_info = extend_mhd_info(mhd_run)

###########################################
## GLOBAL ENERGY SPECTRUM - MULTIPLE ION ##
###########################################

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

def plot_global_spectra_2(sde_run_config,sde_run_config_1,sde_run_config_2,sde_run_config_3):
    """Plot global spectra

    Arguments:
        sde_run_config (dict): SDE run configuration
    """
    kwargs = {"color": 'black', "show_plot": False, "plot_power": False, "power_test": False,"label_text": r"Protons"}
    kwargs_1 = {"color": 'red', "show_plot": False, "plot_power": False, "power_test": False,"label_text": r"Helium"}
    kwargs_2 = {"color": 'blue', "show_plot": False, "plot_power": False, "power_test": False, "label_text": r"Oxygen"}
    kwargs_3 = {"color": 'green', "show_plot": False, "plot_power": False, "power_test": False,"label_text": r"Iron"}

    # First simulation
    tmin = sde_run_config["tmin"]
    tmax = sde_run_config["tmax"]
    run_name = sde_run_config["run_name"]
    spl = run_name.split("/")
    mhd_run = spl[0]
    sde_run = spl[1]
    
    # Second simulation
    tmin_1 = sde_run_config_1["tmin"]
    tmax_1 = sde_run_config_1["tmax"]
    run_name_1 = sde_run_config_1["run_name"]
    spl_1 = run_name_1.split("/")
    mhd_run_1 = spl_1[0]
    sde_run_1 = spl_1[1]
    
    # Third simulation
    tmin_2 = sde_run_config_2["tmin"]
    tmax_2 = sde_run_config_2["tmax"]
    run_name_2 = sde_run_config_2["run_name"]
    spl_2 = run_name_2.split("/")
    mhd_run_2 = spl_2[0]
    sde_run_2 = spl_2[1]
    
    # Fourth simulation
    tmin_3 = sde_run_config_3["tmin"]
    tmax_3 = sde_run_config_3["tmax"]
    run_name_3 = sde_run_config_3["run_name"]
    spl_3 = run_name_3.split("/")
    mhd_run_3 = spl_3[0]
    sde_run_3 = spl_3[1]

    rect = [0.16, 0.16, 0.8, 0.8]
    fig1 = plt.figure(figsize=[3.5, 2.5])
    ax1 = fig1.add_axes(rect)
    
    ### Protons ###
    for tframe in range(tmin, tmax + 1):
        fname = "../../" + run_name + "/fdists_" + str(tframe).zfill(4) + ".h5"
        with h5py.File(fname, "r") as fh:
            pbins_edges = fh["pbins_edges_global"][:]
            fdist = fh["fglobal"][:, :]
        dnptl_bins = np.sum(fdist, axis=1) 
        pbins = 0.5 * (pbins_edges[1:] + pbins_edges[:-1])
        if tframe == tmin:
            ebins_kev, _, debins_kev = get_ebins_kev(sde_run_config, pbins_edges)
        fe = dnptl_bins / debins_kev 
        flux = dnptl_bins * pbins / debins_kev
        kwargs["plot_power"] = False
        plot_energy_spectrum(ebins_kev, flux, ax1, sde_run_config, tframe, **kwargs)

    ### Helium ###
    for tframe in range(tmin_1, tmax_1 + 1):
        fname = "../../" + run_name_1 + "/fdists_" + str(tframe).zfill(4) + ".h5"
        with h5py.File(fname, "r") as fh:
            pbins_edges_1 = fh["pbins_edges_global"][:]
            fdist_1 = fh["fglobal"][:, :]
        dnptl_bins_1 = np.sum(fdist_1, axis=1) 
        pbins_1 = 0.5 * (pbins_edges_1[1:] + pbins_edges_1[:-1])
        if tframe == tmin_1:
            ebins_kev_1, _, debins_kev_1 = get_ebins_kev(sde_run_config_1, pbins_edges_1)
        fe_1 = dnptl_bins_1 / debins_kev_1 
        flux_1 = dnptl_bins_1 * pbins_1 / debins_kev_1
        kwargs_1["plot_power"] = False
        plot_energy_spectrum(ebins_kev_1, flux_1, ax1, sde_run_config, tframe, **kwargs_1)

    ### Oxygen ###
    for tframe in range(tmin_2, tmax_2 + 1):
        fname = "../../" + run_name_2 + "/fdists_" + str(tframe).zfill(4) + ".h5"
        with h5py.File(fname, "r") as fh:
            pbins_edges_2 = fh["pbins_edges_global"][:]
            fdist_2 = fh["fglobal"][:, :]
        dnptl_bins_2 = np.sum(fdist_2, axis=1) 
        pbins_2 = 0.5 * (pbins_edges_2[1:] + pbins_edges_2[:-1])
        if tframe == tmin_2:
            ebins_kev_2, _, debins_kev_2 = get_ebins_kev(sde_run_config_2, pbins_edges_2)
        fe_2 = dnptl_bins_2 / debins_kev_2
        flux_2 = dnptl_bins_2 * pbins_2 / debins_kev_2
        kwargs_2["plot_power"] = False
        plot_energy_spectrum(ebins_kev_2, flux_2, ax1, sde_run_config, tframe, **kwargs_2)
    
    ### Iron ###
    for tframe in range(tmin_3, tmax_3 + 1):
        fname = "../../" + run_name_3 + "/fdists_" + str(tframe).zfill(4) + ".h5"
        with h5py.File(fname, "r") as fh:
            pbins_edges_3 = fh["pbins_edges_global"][:]
            fdist_3 = fh["fglobal"][:, :]
        dnptl_bins_3 = np.sum(fdist_3, axis=1) 
        pbins_3 = 0.5 * (pbins_edges_3[1:] + pbins_edges_3[:-1])
        if tframe == tmin_3:
            ebins_kev_3, _, debins_kev_3 = get_ebins_kev(sde_run_config_3, pbins_edges_3)
        fe_3 = dnptl_bins_3 / debins_kev_3
        flux_3 = dnptl_bins_3 * pbins_3 / debins_kev_3
        kwargs_3["plot_power"] = False
        plot_energy_spectrum(ebins_kev_3, flux_3, ax1, sde_run_config, tframe, **kwargs_3)   
        
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.set_xlabel(r'$\varepsilon$ (keV) / nucleon', fontsize=10)
    ax1.set_ylabel(r'$J$', fontsize=10)
    ax1.tick_params(which='both',labelsize=8,direction='in')
    ax1.grid(True)
    ax1.set_xlim([0.01, 500])
    ax1.set_ylim([0.001,100000000])
    ax1.legend(loc="upper right",prop={'size': 5.5},labelspacing=0.5,ncol =1)
    
    fdir = "../img/global_spectrum/" + mhd_run + "/"
    mkdir_p(fdir)
    fname = fdir + "espect_" + sde_run + ".jpg"
    fig1.savefig(fname, dpi=400)
    plt.show()
    
    emax = []
    emax.append(ebins_kev_3[90]/ebins_kev[70])
    emax.append(ebins_kev_2[84]/ebins_kev[70])
    emax.append(ebins_kev_1[76]/ebins_kev[70])
    emax.append(ebins_kev[70]/ebins_kev[70])
    
    qm_ratio = []
    qm_i = 14.0/56.0
    qm_ratio.append(qm_i)
    qm_o = 6.0/16.0
    qm_ratio.append(qm_o)
    qm_h = 1.0/2.0
    qm_ratio.append(qm_h)
    qm_p = 1.0/1.0
    qm_ratio.append(qm_p)
    
    y_err = []
    deltaxp2 = np.power(0.5*(ebins_kev[71]-ebins_kev[70])/ebins_kev[70],2)
    deltaxhe2 = np.power(0.5*(ebins_kev_1[77]-ebins_kev_1[76])/ebins_kev_1[76],2)
    deltaxo2 = np.power(0.5*(ebins_kev_2[85]-ebins_kev_2[84])/ebins_kev_2[84],2)
    deltaxfe2 = np.power(0.5*(ebins_kev_3[91]-ebins_kev_3[90])/ebins_kev_3[90],2)
    deltay2 = np.power(0.5*(ebins_kev[71]-ebins_kev[70])/ebins_kev[70],2)
    
    errp = (ebins_kev[70]/ebins_kev[70])*np.sqrt(deltaxp2 + deltay2)
    errhe = (ebins_kev_1[76]/ebins_kev[70])*np.sqrt(deltaxhe2 + deltay2)
    erro = (ebins_kev_2[84]/ebins_kev[70])*np.sqrt(deltaxo2 + deltay2)
    errfe = (ebins_kev_3[90]/ebins_kev[70])*np.sqrt(deltaxfe2 + deltay2)
    y_err.append(errfe)
    y_err.append(erro)
    y_err.append(errhe)
    y_err.append(errp)
    
    fig2 = plt.figure(figsize=[3.5, 2.5])
    rect2 = [0.2, 0.16, 0.79, 0.8]
    ax2 = fig2.add_axes(rect2)
    plt.errorbar(qm_ratio,emax,yerr = y_err,fmt ='o',color='red',markersize=3,elinewidth=1,linestyle='dotted')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    m, b, r, p, err = linregress(np.log10(qm_ratio),np.log10(emax))
    print(m,err)

    coefs = np.polyfit(np.log(qm_ratio), np.log(emax), 1)
    pred_f = coefs[1] + np.multiply(sorted(np.log(qm_ratio)), coefs[0])
    ax2.plot(sorted(qm_ratio), np.exp(pred_f), 'k--')
    print(coefs)

    plt.xlabel(r'Q$_{X}$/M$_{X}$',fontsize=10)
    plt.ylabel(r'E$_{X,max}$/E$_{H,max}$',fontsize=10)
    ax2.set_xlim([0.22, 1.1])
    ax2.set_ylim([0.15,1.1])
    ax2.set_xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], ['',r'0.4','',r'0.6','',r'0.8','',r'1.0'])
    ax2.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], [r'0.2','',r'0.4','',r'0.6','',r'0.8','',r'1.0'])
    plt.tick_params(direction='in',which='both',labelsize=8)
    plt.savefig('scaling.jpg', dpi=400)
    plt.show()
    
# Load the configuration for SDE runs
with open('spectrum_config.json', 'r') as file_handler:
    config = json.load(file_handler)
with open('spectrum_config_1.json', 'r') as file_handler:
    config_1 = json.load(file_handler)
with open('spectrum_config_2.json', 'r') as file_handler:
    config_2 = json.load(file_handler)
with open('spectrum_config_3.json', 'r') as file_handler:
    config_3 = json.load(file_handler)
    
sde_run = "transport_H_2"
run_name = "athena_reconnection/" + mhd_run + "/" + sde_run
sde_run_config = config[run_name]
p0 = 0.1
e0 = sde_run_config["e0"] / (0.5*p0**2)
sde_run_config["tmin"] = 80
sde_run_config["tmax"] = 80

sde_run_1 = "transport_He"
run_name_1 = "athena_reconnection/" + mhd_run + "/" + sde_run_1
sde_run_config_1 = config_1[run_name_1]
p0_1 = 0.1
e0_1 = sde_run_config_1["e0"] / (0.5*p0_1**2)
sde_run_config_1["tmin"] = 80
sde_run_config_1["tmax"] = 80

sde_run_2 = "transport_O"
run_name_2 = "athena_reconnection/" + mhd_run + "/" + sde_run_2
sde_run_config_2 = config_2[run_name_2]
p0_2 = 0.1
e0_2 = sde_run_config_2["e0"] / (0.5*p0_2**2)
sde_run_config_2["tmin"] = 80
sde_run_config_2["tmax"] = 80

sde_run_3 = "transport_Fe"
run_name_3 = "athena_reconnection/" + mhd_run + "/" + sde_run_3
sde_run_config_3 = config_3[run_name_3]
p0_3 = 0.1
e0_3 = sde_run_config_3["e0"] / (0.5*p0_3**2)
sde_run_config_3["tmin"] = 80
sde_run_config_3["tmax"] = 80

plot_global_spectra_2(sde_run_config,sde_run_config_1,sde_run_config_2,sde_run_config_3)

