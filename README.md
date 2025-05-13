# covariance-matrix
Code and data associated with the paper: The covariance matrix of metapopulation disease models and applications to early warning signals

Primary authors (contributed equally): Joshua Looker.
Other authors (contributed equally): Kat Rock, Louise Dyson
Corresponding author email address: joshua.looker@warwick.ac.uk

## Running the data-driven analysis
The `covid_pre_processing_*.ipynb` notebooks contain pre-processing code to process the census and UKHSA-case data for usage in the data-driven EWS analysis.

The `covid_eigs_full.ipynb` notebook contains most of the final results used in the paper (with some supplementary results also included in the `supp_figures.ipynb` notebook). All results can be found in the `Figures` and `Results` directories.

## Simulation/Theoretical analysis
Code to run the simulations can be found in `simulation_cluster.py` and `sim.sbatch` (note that these were run on a high-performance computing cluster) and to produce the plots can all be found in the `simulations_aggregation_cluster_relative.ipynb` notebook.

## Further figures
Further figures are produced through the `intro_plots.ipynb`, `methods_diagram.ipynb` and `simulations_additional.ipynb` notebooks.

## Data sources
The data sources used in `covid_pre_processing_*.ipynb` are not included (due to size constraints) in this repository, but can be found at links in the accompanying paper (or on the ONS and UKHSA websites).
