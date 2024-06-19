"""
Analysis functions for SDE calculation.
We use SI units here
"""

import math

LIGHT_SPEED = 299792458.0


def calc_ptl_speed(eth, species, verbose=True):
    """ calculate thermal speed

    Args:
        eth: thermal energy in keV
        species: particle species
    """
    if species == 'e':
        pmass = 9.10938356E-31  # in kilogram
        E0 = 1.0 
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "electron"
    elif species == 'ox':
        pmass = 2.6566962E-26  # in kilogram
        E0 = 16.0 # atomic number
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "oxygen"
    elif species == 'fe':
        pmass = 9.2732796E-26  # in kilogram
        E0 = 56.0 # atomic number
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "iron"
    elif species == 'he':
        pmass = 6.6464731E-27  # in kilogram
        E0 = 4.0 # atomic number
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "helium"
    else:
        pmass = 1.6726219E-27  # in kilogram
        E0 = 1.0 # atomic number
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "proton"
    rest_energy = pmass * LIGHT_SPEED**2
    gamma = (eth) * 1E3 * UNIT_CHARGE / rest_energy + 1
    vth = math.sqrt(1.0 - 1.0 / gamma**2)  # in m/s
    if verbose:
        print("Thermal speed for %0.2f keV %s: %fc" % (eth, ptl_name, vth))
    return vth


def calc_va_si(b0, n0,  verbose=False):
    """Calculate the Alfven speed

    Args:
        b0: magnetic field strength in Gauss
        n0: particle number density in cm^-3
    """
    pmass = 1.6726219E-27  # proton mass in kilogram
    mu0 = 4 * math.pi * 1E-7
    va = b0 * 1E-4 / math.sqrt(mu0 * n0 * 1E6 * pmass)
    print("The Alfven speed is %f" % va)
    return va


def calc_va_cgs(n, B):
    """Calculate the Alfven speed

    Args:
        n: density (cm^-3)
        B: magnetic field (Gauss)
    """
    mi = 1.6726219E-24  # proton mass in gram
    va = B / math.sqrt(4 * math.pi * n * mi)  # CGS in cm/s
    return va


def calc_gyro_frequency(bgauss, species, verbose=False):
    """Calculate particle gyro-frequency

    Args:
        bgauss: magnetic field in Gauss
        species: particle species
    """
    if species == 'e':
        pmass = 9.10938356E-31  # in kilogram
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "electron"
    elif species == 'ox':
        pmass = 2.6566962E-26  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 7
        ptl_name = "oxygen"
    elif species == 'fe':
        pmass = 9.2732796E-26  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 14
        ptl_name = "iron"
    elif species == 'he':
        pmass = 6.6464731E-27  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 2
        ptl_name = "helium"
    else:
        pmass = 1.6726219E-27  # in kilogram
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "proton"
    omega = UNIT_CHARGE * bgauss * 1E-4 / pmass
    if verbose:
        print("The gyro-frequency of %s in %0.1f Gauss magnetic field: %e Hz" %
              (ptl_name, bgauss, omega))
    return omega


def calc_kappa_parallel(eth,
                        bgauss,
                        lcorr,
                        knorm,
                        species,
                        va=1.0,
                        L0=1.0,
                        sigma2=1.0):
    """Calculate parallel diffusion coefficient

    Args:
        eth: thermal energy in keV
        bgauss: magnetic field in Gauss
        lcorr: correlation length in km
        knorm: kappa normalization in m^2/s
        species: particle species
        va: Alfven speed in m/s
        L0: length scale in m
        sigma2: turbulence variance
    """
    csc_3pi_5 = 1.05146
    csc_2pi_3 = 1.15470053838
    csc_3pi_4 = 1.41421356237
    csc_pi = 1.0
    lcorr *= 1000  # in meter
    if species == 'e':
        pmass = 9.10938356E-31  # in kilogram
        E0 = 1.0 # atomic number
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "electron"
    elif species == 'ox':
        pmass = 2.6566962E-26  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 7
        E0 = 16.0
        ptl_name = "oxygen"
    elif species == 'fe':
        pmass = 9.2732796E-26  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 14
        E0 = 56.0
        ptl_name = "iron"
    elif species == 'he':
        pmass = 6.6464731E-27  # in kilogram
        UNIT_CHARGE = 1.6021765E-19 * 2
        E0 = 4.0
        ptl_name = "helium"   
    else:
        pmass = 1.6726219E-27  # in kilogram
        E0 = 1.0 # initial energy in keV
        UNIT_CHARGE = 1.6021765E-19
        ptl_name = "proton"
    vth = calc_ptl_speed(eth, species) * LIGHT_SPEED  # in m/s
    print("Magnetic field: %0.3f G" % bgauss)
    print(ptl_name + " energy/nucleon: %0.2f keV" % (eth / E0))
    print("Thermal speed: %e km/s" % (vth / 1E3))
    print("Thermal speed/va: %f" % (vth / va))
    eva = 0.5 * pmass * va**2 / UNIT_CHARGE / 1000  # in kev
    print("Particle energy with speed va: %f keV" % (eva))
    ethermal = 0.5 * pmass * vth**2 / UNIT_CHARGE / 1000  # in kev
    print("Particle energy with thermal speed: %f keV" % (ethermal))
    omega = calc_gyro_frequency(bgauss, species)
    rg = vth / omega  # gyro-radius in m
    
    # Kolmogorov model for turbulence
    kpara = 3 * vth**3 * csc_3pi_5 / (20 * lcorr * omega**2 * sigma2)
    kpara *= 1.0 + (72.0 / 7) * (omega * lcorr / vth)**(5 / 3)
    
    # IK model for turbulence
    #kpara = vth**3 * csc_2pi_3 / (6 * lcorr * omega**2 * sigma2)
    #kpara *= 1.0 + (32.0 / 5) * (omega * lcorr / vth)**(3 / 2)
    
    # Random model for turbulence
    #kpara = vth**3 * csc_pi / (4 * lcorr * omega**2 * sigma2)
    #kpara *= 1.0 + (8 / 3) * (omega * lcorr / vth)
    
    # New random model for turbulence
    #kpara = 3 * vth**3 * csc_3pi_4 / (16 * lcorr * omega**2 * sigma2)
    #kpara *= 1.0 + (9.0 / 2.0) * (omega * lcorr / vth)**(4 / 3)
    
    kperp = (5 * vth * lcorr / 12) * math.sin(3 * math.pi / 5) * sigma2
    gamma = 1.0 / math.sqrt(1 - (vth / LIGHT_SPEED)**2)
    p = gamma * pmass * vth
    v1 = UNIT_CHARGE * va / (p * LIGHT_SPEED)
    v2 = pmass * UNIT_CHARGE * va / p**2
    v1 *= bgauss * 1E-4 * L0
    v2 *= bgauss * 1E-4 * L0
    tau0 = 3 * kpara / vth**2  # scattering time scale
    tau0_scattering = tau0 / (L0 / va)
    print("Gyro-frequency : %e Hz" % omega)
    print("Gyro-radius: %e m" % rg)
    print("Mean-free path: %e m" % (tau0 * vth))
    print("kappa parallel: %e m^2/s" % kpara)
    print("Normed kappa parallel: %e" % (kpara / knorm))
    print("kperp / kpara: %e" % (kperp / kpara))
    print("Parameters for particle drift: %e, %e" % (v1, v2))
    print("rg*vth/L0: %e" % (rg * vth / L0))
    print("Scattering time for initial particles: %e" % tau0_scattering)
    print("Mean free path initial particles: %e\n" % (tau0 * vth))

def reconnection_test():
    """parameters for reconnection test
    """
    
    L0 = 5.0E9 # 5.0E6  # in m
    b0 = 0.002 # 50.0  # in Gauss
    n0 = 1.5E3 # 1.0e10  # number density in cm^-3
    lscale = 1.0  # scale the MHD length
    bscale = 1.0  # scale the magnetic field
    slab_component = 1.0
    L0 *= lscale
    b0 *= bscale
    n0 *= bscale**2  # to keep the Alfven speed unchanged
    va = calc_va_cgs(n0, b0) / 1E2  # in m/s
    knorm = L0 * va
    Lc = 5.0E4 # 333  # in km
    sigma2 = slab_component
    print("Alfven speed: %0.1f km/s" % (va / 1E3))
    print("Correlation length: %0.1f km" % Lc)
    print("Turbulence variance: %e\n" % sigma2)
    calc_kappa_parallel(5, b0, Lc, knorm, 'p', va, L0, sigma2)


if __name__ == "__main__":
    reconnection_test()
