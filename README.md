## Description

A structural model to estimate the effect of compulsory military service on education and income of individuals.

This repository is the replication code of the *Hoseini and Sabouri (2021)* working paper:

"The Effect of Compulsory Military Service on Lifetime Income of Men in Iran, A Structural Model Estimation"

More information and manuscript of the paper can be found [here](asfddf).

## Notes on replication
This repository is under development. However, description of the codes are as follows:

* *Main_CPU.jl* : solve and simulate the model using parallel computation on the CPU

* *Main_GPU.jl* : do the same calculation as the Main_CPU.jl code but more efficient and faster on the GPU.

* *Main_Approximation.jl* : solve the model with the approximation developed in Keane and Wolpin (1994). This code is not used in the paper and is not complete yet.

After solving and simulating the model, an array of the individuals characteristcis and choice will be generate and save on the local memory.
This will be serve as the *model simulated data* in the Python code to caclulate the counterfactual and estimating the results.
