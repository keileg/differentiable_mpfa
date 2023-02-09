# Runscripts and similar for developing differentiable Mpfa/Tpfa 
This repository contains runscripts and utility tools used for the development of the manuscript *Numerical treatment of variable permeability in multiphysics problems*,
soon to be uploaded on arXiv.

To reproduce the results reported in this work, you need PorePy (https://github.com/pmgbergen/porepy) installed on your system and set to a reasonable commit
(the results reported in the paper was produced on commit id b1122156c5677025d939f7c8a12c3472a8e8a8c2). The simulations reported in the manuscript can be reproduced
by the four scripts `run_verification.py`, `run_chemistry.py` `run_biot_matrix.py` and `run_biot_fracture.py`.

To be sure that the results in the manuscript are truly reproduced, we recommend using the Docker container (soon to be made) available at https://doi.org/10.5281/zenodo.7624095.
