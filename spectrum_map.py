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
    # Normalizations
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
    
    # Get the time intervals for different outputs
    for i in range(1, 100):
        vname = "output" + str(i)
        if vname in mhd_info:
            dtv = "dt_" + mhd_info[vname]["file_type"]
            mhd_info[dtv] = mhd_info[vname]["dt"] * mhd_info["norm"]["t0"] # normalization of time outputs in seconds
        else:
            break
    mhd_info.update(mhd_config_for_sde)
    return mhd_info
    
mhd_run = "beta_0.5_2"
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

##################################
## DISTRIBUTION OF ENERGY BANDS ##
##################################

def spatial_distribution(plot_config, mhd_info, sde_run_config, verbose=True, show_plot=True):
    """Plot spatial distribution for multi energy bands

    Arguments:
        plot_config (dict): plot configuration in dictionary
        mhd_info (dict): MHD simulation information
    """
    tframe = plot_config["tframe"]
    ng = mhd_info["nghost"]
    ## nghost = mhd_info["nghost"]
    nx_mhd = mhd_info["nx"]  # r
    ny_mhd = mhd_info["ny"]  # theta
    nxg = nx_mhd + 2 * ng # add the ghost cells
    nyg = ny_mhd + 2 * ng
    
    ## dx = mhd_info["dx"]
    ## dy = mhd_info["dy"]
    ## bx = mhd_info["bcx"]
    ## by = mhd_info["bcy"]

    fig = plt.figure(figsize=[6.5, 3])
    gs = gridspec.GridSpec(1,
                           5,
                           wspace=0,
                           hspace=0,
                           top=0.92,
                           bottom=0.14,
                           left=0.08,
                           right=0.98)
    sizes = [
        mhd_info["xmin"],
        mhd_info["xmax"],
        mhd_info["ymin"],
        mhd_info["ymax"]
    ]
    
    # This normalizes the sizes of the domain (plus includes the conversion in Mm)
    L0_Mm = mhd_info["norm"]["L0"] / 1E6
    sizes = np.asarray(sizes) * L0_Mm
    lx = sizes[1] - sizes[0] # dimension of x in Mm
    ly = sizes[3] - sizes[2] # dimension of y in Mm
    xmid = 0.5 * (sizes[0] + sizes[1])
    xmin_in = sizes[0] # +7.5 # The factor 0.5 is needed for the case with double current sheet
    xmax_in = sizes[1] # -7.5
    ymin_in = sizes[2]        # These are the starting/ending points of the domain
    ymax_in = sizes[3]
    
    print(xmin_in, xmax_in)
    
    ixs = int(xmin_in * nx_mhd / lx) # multiply by the number of cells in one direction and divide by the size in Mm
    ixe = int(xmax_in * nx_mhd / lx)
    iys = int(ymin_in * ny_mhd / ly)
    iye = int(ymax_in * ny_mhd / ly)
    mhd_run_dir = mhd_info["run_dir"]
    fpath = mhd_run_dir + 'bin_data/'
    xmid = (xmin_in + xmax_in) / 2 # middle point along x
    sizes_in = [xmin_in-xmid, xmax_in-xmid, ymin_in, ymax_in]
    
    # Here below we have the part relative to the division into energy bands
    
    fnorms = [1, 1E1, 1E2, 1E3] # Intervals of energy (multiples of n) showed in the plot
    
    tframe_str = str(tframe).zfill(4)
    run_name = sde_run_config["run_name"] # run_name is the run directory
    fname = '../../' + run_name + "/fdists_" + str(tframe).zfill(4) + ".h5"
    with h5py.File(fname, "r") as fh:
        pbins_edges = fh["pbins1_edges"][:]
        mubins_edges = fh["mubins1_edges"][:]
        fdist = fh["flocal1"][:, :, :, :, :]
    nzr, nyr, nxr, npbins, nmu = fdist.shape
    rx = nx_mhd // nxr
    ry = ny_mhd // nyr
    ixsr = int(xmin_in * nxr / lx) # multiply by the reduced nr. of cells and divide by the size in Mm
    ixer = int(xmax_in * nxr / lx)
    iysr = int(ymin_in * nyr / ly)
    iyer = int(ymax_in * nyr / ly)
    nptl_tot = np.sum(fdist)
    if verbose:
        print("Total number of particles: %f" % np.sum(fdist))

    _, ebins_edges_kev, _ = get_ebins_kev(sde_run_config, pbins_edges)
    fband_ycuts = []
    fband_xcuts = []
    
    bands, bande = plot_config["high_bands"]
    for iband, eband in enumerate(range(bands, bande + 1)):
        fband = np.sum(np.sum(fdist[:, :, :, eband, :], axis=3), axis=0)
        if verbose:
            nptl_band = np.sum(fband)
            print("particle number in band %d: %f, %0.2f%%" % (eband, nptl_band, nptl_band*100/nptl_tot))
            print("min, max, mean, and std: %f %f %f %f" % (np.min(fband), np.max(fband), np.mean(fband), np.std(fband)))
            
        fband *= fnorms[iband]
        nyl, nxl = fband.shape
        fdata_vcut = np.mean(fband[:, nxl // 4 - 3:nxl // 4 + 4], axis=1)
        fband_ycuts.append(fdata_vcut)
        ax = plt.subplot(gs[0, iband])
        vmin = plot_config['nmin']
        vmax = plot_config['nmax']
        fdata = fband[iysr:iyer, ixsr:ixer] + vmin * 0.1
        
        img = ax.imshow(fdata,
                        cmap=plt.cm.viridis,
                        aspect='equal',
                        origin='lower',
                        extent=sizes_in,
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r'$x$ (Mm)', fontsize=10)
        
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,2))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.get_offset_text().set_fontsize(8)
        ax.yaxis.get_offset_text().set_fontsize(8)
        
        if iband > 0:
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_ylabel(r'$y$ (Mm)', fontsize=10)

        if iband == 0:
            rect = np.asarray(ax.get_position()).flatten()
            rect_cbar = np.copy(rect)
            rect_cbar[0] += 0.01
            rect_cbar[2] = 0.01
            rect_cbar[1] = 0.6
            rect_cbar[3] = 0.2
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(img, cax=cbar_ax, extend="both")
            cbar.ax.tick_params(labelsize=6, color='w')
            cbar.ax.yaxis.set_tick_params(color='w')
            cbar.outline.set_edgecolor('w')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')
        norm = fnorms[iband]
        enel = "{%0.0f}" % (ebins_edges_kev[eband])
        eneh = "{%0.0f}" % (ebins_edges_kev[eband + 1])
        ftext = (enel + ' -- ' + eneh + 'keV')
        if norm > 1:
            fig_text = r'$' + str(int(norm)) + 'n($' + ftext + r'$)$'
        else:
            fig_text = r'$n($' + ftext + r'$)$'
        #cbar.ax.set_ylabel(fig_text, fontsize=12, color='w')
        ax.text(0.5,
                0.95,
                fig_text,
                color=tableau_colors[iband],
                fontsize=6,
                bbox=dict(boxstyle='round',
                          facecolor='w',
                          alpha=1,
                          edgecolor='none',
                          pad=0.2),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        
    # Here below there is the part relative to the density
    ax = plt.subplot(gs[0, 4])
    xgrid = np.linspace(sizes[0], sizes[1], nxr)
    ygrid = np.linspace(sizes[2], sizes[3], nyr)
    for iband, fband_y in enumerate(fband_ycuts):
        label = 'band ' + str(iband + 1)
        ax.semilogx(fband_y,
                    ygrid,
                    nonpositive="mask",
                    linewidth=1,
                    label=label)
    ax.set_ylim([ymin_in, ymax_in])
    ax.set_xlim([1E0, 2E3])
    ax.grid(linewidth=0.5)
    ax.set_xlabel('Density', fontsize=10)
    ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(labelsize=8)

    tva = mhd_info["dt_hdf5"] * tframe
    tva1 = mhd_info["dt_out"] * tframe
    title = r'$t = ' +str(round(tva1,2))+ r' t_{A}$'#+', total number of particles: %f' % np.sum(dists)
    # title = r'$t = ' + "{:10.1f}".format(tva) + r' s$' # Normalized time title
    plt.suptitle(title, fontsize=12)

    fdir = '../img/' + mhd_run + '/' + sde_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'nrho_vertical_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)
    
    print(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')
        
# Load the configuration for SDE runs
with open('spectrum_config.json', 'r') as file_handler:
    config = json.load(file_handler)
    
sde_run = "transport_H_0"
run_name = "athena_reconnection/" + mhd_run + "/" + sde_run
sde_run_config = config[run_name]

plot_config = {}
#if "high_bands" in sde_run_config:
#plot_config["high_bands"] = sde_run_config["high_bands"]
#else:
plot_config["high_bands"] = [3, 6]
plot_config["eband"] = 3
plot_config["tframe"] = 30
plot_config["nmin"] = 5E-8
plot_config["nmax"] = 5E2

spatial_distribution(plot_config, mhd_info, sde_run_config, verbose=True, show_plot=True)
