# covariance-matrix
Code and data associated with the paper: The covariance matrix of metapopulation disease models and applications to early warning signals

Primary author: Joshua Looker.
Other authors (contributed equally): Kat Rock, Louise Dyson
Corresponding author email address: joshua.looker@warwick.ac.uk

## Running the data-driven analysis
The `covid_pre_processing_*.ipynb` notebooks contain pre-processing code to process the census and UKHSA-case data for usage in the data-driven EWS analysis.

The `covid_eigs.ipynb` and `sir_results_*.ipynb` notebooks contains most of the final results used in the paper (with some supplementary results also included in the `supp_figures.ipynb` and `sir_null_model.ipynb` notebooks). All figures can be found in the `Rev_Figures` and `Rev_Figures_Small` directories.

## Simulation/Theoretical analysis
Code to run the simulations can be found in `meta_sim_*.py`, `sim_cluster_small.py` and `sim.sbatch` (note that these were run on a high-performance computing cluster).

## Data sources
The data sources used in `covid_pre_processing_*.ipynb` are not included (due to size constraints) in this repository, but can be found at links in the accompanying paper (or on the ONS and UKHSA websites).