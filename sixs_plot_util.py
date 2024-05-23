#!/home/chospa/anaconda3/bin/python

"""
A file that holds useful constants and little helper functions.
"""

import os

# A path pointing to the directory in which the script is run
CURRENT_DIRECTORY = os.getcwd()

# Plotting related constants
FIGSIZE = (13,10)

UPPER_RIGHT = 1
UPPER_LEFT = 2
LOWER_LEFT = 3
LOWER_RIGHT = 4

ELECTRON_ELIMS = (2e-2, 5e1)
PROTON_ELIMS = (5e-1, 5e3)

ELECTRON_GLIMS = (1e-4, 2e-1)
PROTON_GLIMS = (1e-5, 2e-1)

CONTAMINATION_ELIMS = (1e-1, 1e2)
CONTAMINATION_GLIMS = (1e-5, 1e-2)

P_CONTAMINATION_ELIMS = (1e-1, 1e2)
P_CONTAMINATION_GLIMS = (1e-5, 1e-2)

E_CONTAMINATION_ELIMS = (3e-1, 2e1)
E_CONTAMINATION_GLIMS = (1e-6, 5e-2)

FONTSIZES = {
    "title" : 25,
    "axes_labels" : 22,
    "legend" : 18
}

# Instrument channel boundary values 
SIDE0_E1_HIGH_BOUND = 0.06792525070055476

SIDE0_CHANNEL_LOW_BOUNDS = {
    "E1" : 0.050936752167801344,
    "E2" : 0.0679252507005548,
    "E3" : 0.104599895343025,
    "E4" : 0.19282185207892,
    "E5" : 0.567422104279643,
    "E6" : 1.06498563535043,
    "E7" : 15.5383983127497,
    "P1" : 0.991045856248861,
    "P2" : 1.1039991779174,
    "P3" : 1.25214968906556,
    "P4" : 1.82691671794092,
    "P5" : 2.86438407149338,
    "P6" : 6.09756235221459,
    "P7" : 11.039991779174,
    "P8" : 18.2691671794092,
    "P9" : 31.9084898062911
}

UNDESIRED_CROSS_CHANNELS = ("EP5","EP6","EP7","PE4","PE5","PE6")

def set_standard_response_plot_settings(ax):
    """
    Sets axes to log-log scale.
    Sets axis labels to E[MeV] and R [cm^2 sr].
    Sets tick parameters to something I found experimentally to be more aestethically pleasing
    Sets the legend to outside of the plot in the top right corner
    """

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_ylabel(r"R [$\mathrm{cm}^{2}$ sr]", fontsize=FONTSIZES["axes_labels"])
    ax.set_xlabel("E [MeV]", fontsize=FONTSIZES["axes_labels"])

    ax.tick_params(axis="both", which="major", size=10, width=2, labelsize=FONTSIZES["axes_labels"])
    ax.tick_params(axis="both", which="minor", size=6, width=1.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1.,1.02), fontsize=FONTSIZES["legend"], frameon=True, fancybox=True)
