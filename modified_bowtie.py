#!/home/chospa/anaconda3/bin/python

"""
A module for the new modified bowtie analysis.

Based on an earlier implementation by Philipp Oleynik (https://www.utupub.fi/handle/10024/152846 and references therein).

@ Last updated: 2024-03-27
"""

__author__ = "Christian Palmroos"
__credits__ = ["Christian Palmroos", "Rami Vainio", "Philipp Oleynik"]
__status__ = "Development"

# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sixs_plot_util import *


def default_path(side:int):
    """
    Returns the standard path to response stats subdirectory. The path ends with an os.sep.
    """
    return f"{CURRENT_DIRECTORY}{os.sep}side{side}_response_stats{os.sep}"


def save_bowtie_results(results:dict, particle:str=None):
    """
    Saves the bowtie_results dictionary to a .csv table in the current directory.

    @param results : {dict} A dictionary returned by the modified_bowtie() function.
    @param particle : {str} Either 'e' or 'p' for electron or proton- respectively. 
    """

    # If particle is not provided, try to infer it
    if particle is None:
        if "E1" in results:
            particle = "electron"
        elif "P1" in results:
            particle = "proton"
        else:
            raise NameError("Parameter 'particle' not provided, nor could it be inferred.")

    df = pd.DataFrame(results).T

    df = df.rename(columns={0 : "lower_bound",
                            1 : "higher_bound",
                            2 : "geom_factor"})
    
    df.to_csv(f"{particle}_bowtie_results.csv")


def read_response_data(side:int, particle:str='e', path:str=None):
    """
    Reads the response data given by the response matrix that bowtie_calc.py produces.
    
    @param particle : {str}, The identifier for particle species. Either 'e' or 'p'.
    """

    # If path not provided, use the default path
    if path is None:
        path = default_path(side=side)

    errormsg = f"Argument particle (str) has to be either 'e' or 'p', not {particle}"
    
    if not isinstance(particle,str):
        raise TypeError(errormsg)
    if particle not in ('e', 'p'):
        raise ValueError(errormsg)
    
    responses = np.load(f"{path}{particle}_channel_responses.npy")
    energies = np.load(f"{path}{particle}_incident_energies.npy")
    names = np.load(f"{path}{particle}_channel_names.npy")
    
    # Since all energies are identical, that is used as the index
    df = pd.DataFrame(data=responses.T, index=energies[0], columns=names, dtype=float)
    
    return df


def read_channel_bounds(side:int, particle:str, path:str=None):
    """
    Returns the lower and higher energy boundaries of sixs-p channels for 
    the given side.
    """

    # If path was not provided, try the default path
    if path is None:
        path = default_path(side=side)

    filename = f"side{side}_{particle}_channel_boundaries.csv"

    # The dataframe is organized such that for each channel (row) there are
    # lower bound and a higher bound (columns)
    df = pd.read_csv(f"{path}{filename}", index_col="channel")
    
    return df


def modified_bowtie(data, side, channel=None, particle=None, contamination=False, sum_channels=False, plot=True, save=False, max_energy=10):
    """
    Modified bow-tie analysis to produce an effective energy range for an energy channel.
    
    Parameters:
    -----------
    data : {pd.Dataframe}
                Pandas DataFrame object that holds channel responses indexed by incident energy.

    side : {int}
                An integer [0,4] that chooses the side (aka viewing direction) of SIXS-P.

    channel : {str}
                Channel identifier, e.g., 'E1' or 'P5'.

    particle : {str}
                Either 'e' or 'p'. If not provided, try to infer it from DataFrame headers.

    contamination : {bool}
                Runs bowtie-analysis on cross-channels, i.e., electron response function applied on proton channels,
                such as PE1, PE2 and PE3.
                Not implemented yet. 

    sum_channels : {bool}
                Runs bowtie-analysis on summed channels, e.g., 'P1+P2' etc, instead of normal channels.
                Not implemented yet.

    plot : {bool}
                Plots the bow-tie analysis, and the response function with a fitted equivalent box-car.
    
    save : {bool}
                A save-switch to save the plots in the current directory.

    max_energy : {int, float}
                The maximum energy to integrate to.

    Returns:
    --------
    result : {dict}
                Includes the lower bound, the upper bound and the constant geometric factor for each
                channel. Upper bound and geom. factor replaced with nan if not found.
    """

    # @TODO:
    # Implement cross-channel bowtie-analysis, i.e., proton channel response as a 
    # function of electron energy.
    if contamination or sum_channels:
        raise NotImplementedError(f"Parameters 'contamination' and 'sum_channels' are not implemented yet!")

    if particle is None:
        try:
            particle = 'e' if 'E' in data.columns[0] else 'p'
        except ValueError as vale:
            print(vale)
            return None

    # Choose channel or all channels
    if channel is None:
        channels = [data.columns.to_list()[0]]
        print(f"No channel provided, assuming the first channel {channels[0]}.")
    elif isinstance(channel,str):
        if channel == "all":
            channels = data.columns.to_list()
        else:
            channels = [channel]
    elif isinstance(channel,(list,tuple)):
        channels = channel
    else:
        raise TypeError(f"The argument 'channels' needs to be a string or a list of strings, not {type(channel)}.")

    # The x-axis of the plots, incident energy of the particles
    energy_ticks = data.index.values # np.linspace(0.1, MAX_ENERGY, len(data))

    # loop through the channel(s) and collect results to a dictionary
    results = {}
    for channel in channels:

        # Here run modified bow-tie to a single energy channel
        channel_bounds_df = read_channel_bounds(side=side, particle=particle)
        lower_bound = channel_bounds_df["lower_bound"][channel]
        varied_responses, e2_varies, gammas = bowtie_gamma_variance(channel_low_bound=lower_bound, response_function=data[channel].values,
                                                                    energy_ticks=energy_ticks, max_energy=max_energy)

        g_delta, minimum_at = calculate_g_delta(varied_responses)

        # The optimal E2 is at the point at which deltaG comes to its local minimum
        higher_bound = e2_varies[minimum_at]

        # ... And the geometric factor, that is the average of all varied response functions at that point
        bestG = np.nanmean([resp[minimum_at] for resp in varied_responses])

        # If the delta in G decreases monotonically, it means that this function has no local minimum that is not at the
        # end of the interval under consideration. Hence, there is no optimal E2 and no corresponding bestG
        if g_delta.is_monotonic_decreasing:
            boxcar = (lower_bound, np.nan, np.nan)
        else:
            boxcar = (lower_bound, higher_bound, bestG)

        if plot:
            # Plots the bow-tie analysis for a channel
            plot_g_vs_e(varied_responses=varied_responses, energy_ticks=e2_varies, 
                        gamma_varies=gammas, channel=channel,
                        g_delta=g_delta, g_delta_min=higher_bound, save=save)

            # Plots the response function of a channel
            plot_single_response_function(energies=energy_ticks, response=data[channel].values, 
                                          channel=channel, boxcar=boxcar, save=save)

        results[channel] = boxcar
        
    return results


def calculate_g_delta(varied_responses):
    """
    Calculates the difference between the highest and lowest value of G_gamma(E_2) at each energy E2

    Returns:
    g_delta_series : {pd.Series}

    g_delta_series.idxmin() : {int}
    """
    
    g_delta = np.zeros(len(varied_responses[0]))

    # A matrix of modified response functions as a function of E_2
    response_matrix = np.array(varied_responses).T
    
    for i in range(len(g_delta)):
        ming, maxg = np.nanmin(response_matrix[i]), np.nanmax(response_matrix[i])
        g_delta[i] = maxg - ming

    g_delta_series = pd.Series(g_delta)

    return g_delta_series, g_delta_series.idxmin()


def bowtie_gamma_variance(channel_low_bound, response_function, energy_ticks, gamma_varies=None, delta_e_varies=None,
                          max_energy=10):
    """
    
    Parameters:
    -----------
    channel_low_bound : {float}
                        The lower bound of an energy channel in MeV

    response_function : {array}
                        The geometric factor of the instrument as a function of energy

    energy_ticks  : {array}
                        The energies corresponding to the response function's values in MeVs

    gamma_varies : {array, list}
                        A range of different values for the spectral index of a particle spectrum

    delta_e_varies : {array, list}
                        A range of different widths for the energy channel to try
    """

    if gamma_varies is None:
        gamma_varies = np.linspace(1.5, 4.0, 11)

    # By default let's vary the width of the channel from 0.001 MeV (1 keV) to 5 MeV
    if delta_e_varies is None:
        # Numpy logspace takes the start and end EXPONENTS as arguments, and returns logarithmically spaced
        # array of values.
        delta_e_varies = np.logspace(-3, np.log10(max_energy), len(response_function))

    e2_varies = delta_e_varies + channel_low_bound

    # In this list we store different geometric factors as a function of gamma
    modified_response_functions = []
    for gamma in gamma_varies:

        g_gamma = calculate_G(energy_ticks=energy_ticks, gamma=gamma, response=response_function, 
                              channel_low_bound=channel_low_bound, e2_varies=e2_varies)
        
        modified_response_functions.append(g_gamma)
    
    return modified_response_functions, e2_varies, gamma_varies


def calculate_G(energy_ticks, gamma, response, channel_low_bound, e2_varies):

    # This is the nominator of the equation, the integrand of the counting rate equation without
    # the constant A.
    integrand = np.power(energy_ticks, -gamma) * response

    integral = np.trapz(y=integrand, x=energy_ticks)

    nominator = (gamma - 1) * integral

    # These are the denominators of the modified bow-tie analysis equation for the geometric factor
    low_bound_limit = np.power(channel_low_bound, 1-gamma)

    denominators = low_bound_limit - np.power(e2_varies, 1-gamma)

    g_gamma = nominator / denominators
    
    return g_gamma


def plot_g_vs_e(varied_responses, energy_ticks, gamma_varies, channel=None, g_delta=None, g_delta_min=None, save=False):
    """
    Plots a range of response functions given different spectral indices.
    """

    if g_delta is not None:
        if isinstance(g_delta, pd.Series):
            g_delta = g_delta.values
    
    if channel is None:
        chnl_str = "Channel"
    else:
        chnl_str = channel

    colors = plt.get_cmap("plasma", 31+1)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_title(f"{chnl_str} response as a function of E$_{2}$ for different $\gamma$", fontsize=FONTSIZES["title"])
    ax.set_yscale("log")
    
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis="both", which="major", size=10, width=2, labelsize=FONTSIZES["axes_labels"])
    ax.tick_params(axis="both", which="minor", size=6, width=1.5, labelsize=FONTSIZES["axes_labels"]-5)

    # Labels
    ax.set_xlabel("$E_{2}$ [MeV]", fontsize=FONTSIZES["axes_labels"])
    ax.set_ylabel("G [cm$^2$ sr]", fontsize=FONTSIZES["axes_labels"])

    # Limits
    ax.set_xlim(np.nanmin(energy_ticks), np.nanmax(energy_ticks))
    ax.set_xscale("log")

    # Go through all the geometric factors 
    for i, response in enumerate(varied_responses):

        ax.plot(energy_ticks, response, color=colors(2*i+1), lw=2.0, label=f"{gamma_varies[i]}")

    if g_delta is not None:
        ax.plot(energy_ticks, g_delta, color="black", lw=3.0, label="$\delta$")
    
    if g_delta_min is not None:
        ax.axvline(x=g_delta_min, color="darkgrey", ls="--", lw=2.0, label=f"min($\delta$):\n{np.round(np.nanmin(g_delta),4)}")

    ax.legend(loc=UPPER_LEFT, bbox_to_anchor=(0.99,1.0), fontsize=FONTSIZES["legend"], 
             labelcolor="linecolor", framealpha=0.0, title="$\gamma$:", title_fontsize="xx-large")

    if save:
        fig.savefig(f"{channel}_responses_wgamma.png", facecolor="white", transparent="False", bbox_inches="tight")
    plt.show()


def plot_single_response_function(energies, response, channel, boxcar=None, save=False):
    """
    Plots the response function of a single energy channel.
    
    @param energies, responses : {float} incident energy and channel response
    @param particle : {str} 'e' or 'p'
    @param boxcar : {list,tuple} [start, end, yvalue]
    """

    particle_str = "electron" if "E" in channel else "proton"

    TITLES = (  f"{channel} response as a function of {particle_str} energy",
                "Combined channel response as a function of particle energy",
                "Proton channel response as a function of electron energy"
             )

    sum_channels = False
    contamination = False

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_ylabel(r"R [$\mathrm{cm}^{2}$ sr]", fontsize=FONTSIZES["axes_labels"])
    ax.set_xlabel("E [MeV]", fontsize=FONTSIZES["axes_labels"])
    ax.set_yscale("log")
    ax.set_xscale("log")

    elims = ELECTRON_ELIMS if particle_str=="electron" and not contamination else PROTON_ELIMS if not contamination else CONTAMINATION_ELIMS
    ax.set_xlim(elims)

    glims = ELECTRON_GLIMS if particle_str=="electron" and not contamination else PROTON_GLIMS if not contamination else CONTAMINATION_GLIMS
    ax.set_ylim(glims)

    # We change the fontsize of minor ticks label 
    ax.tick_params(axis="both", which="major", size=10, width=2, labelsize=FONTSIZES["axes_labels"])
    ax.tick_params(axis="both", which="minor", size=6, width=1.5)

    title_index = 0 if not sum_channels and not contamination else 1 if sum_channels and not contamination else 2
    ax.set_title(TITLES[title_index], fontsize=FONTSIZES["title"])

    ax.plot(energies, response, lw=2.5, color="navy")

    # Plotting channel boundaries
    if boxcar is not None and not np.isnan(boxcar[-1]):

        lower_bound = boxcar[0]
        higher_bound = boxcar[1]
        best_G = boxcar[2]

        ax.vlines(x=lower_bound, ymin=1e-7, ymax=best_G, ls="--", color="black")
        ax.vlines(x=higher_bound, ymin=1e-7, ymax=best_G, ls="--", color="black")
        ax.hlines(y=best_G, xmin=lower_bound, xmax=higher_bound, ls="--", color="black", label="Equivalent box-car response")

        fill_selection = (energies >= lower_bound) & (energies <= higher_bound)
        fill_energy_selection = energies[fill_selection]

        lower_g_bound = np.ones(len(fill_energy_selection)) * 1e-7
        best_G_bound = np.ones(len(fill_energy_selection)) * best_G

        # Fills blue up to response, but never more than best_G
        ax.fill_between(fill_energy_selection, lower_g_bound, best_G_bound, fc="navy", alpha=0.1)
        ax.fill_between(fill_energy_selection, response[fill_selection], best_G_bound, where=best_G_bound>response[fill_selection], fc="white")
        #ax.fill_between(fill_energy_selection, lower_g_bound, response[fill_selection], where=best_G_bound>=response[fill_selection], fc="navy", alpha=0.1) # where=X > upper_bound,

        # Fills up red where best_G is larger than response
        ax.fill_between(fill_energy_selection, response[fill_selection], best_G_bound, where=best_G_bound>response[fill_selection], fc="maroon", alpha=0.1)

        ax.legend(loc=UPPER_RIGHT, bbox_to_anchor=(1.0,1.0), fontsize=FONTSIZES["legend"], frameon=True, fancybox=True)
    
    if save:
        fig.savefig(f"{channel}_response.png", facecolor="white", transparent="False", bbox_inches="tight")
    plt.show()


def main():

    print("This module contains the functions used in the Jupyter notebook 'Modified_Bowtie_Emax'.")

if __name__ == "__main__":
    
    main()
