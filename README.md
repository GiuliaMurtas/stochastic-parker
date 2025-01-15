# stochastic-parker
This repository contains analysis routines for GPAT simulations, modified by myself to examine energy spectra of particle acceleration problems. Original files can be found in `https://github.com/xiaocanli/stochastic-parker`.

List of Python scripts, currently working on Purdue Anvil:

- `spectrum_timelapse.py`: this script plots the energy flux distribution $J$ of a given ion population as a function of time.
- `spectrum_multiion.py`: this script compares the energy flux distribution of multiple ions at a given time output.
- `spectrum_map.py`: this script plots the spatial distribution of particles in four different energy bands, and includes the variation of the number density $\rho$ along a vertical slice at the center of the domain. This setup has been prepared for a reconnection problem (current sheet located vertically at the center of the domain).
