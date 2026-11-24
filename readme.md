
# Inventory Planning in Capacitated High-Tech Assembly Systems Under Non-Stationary Demand

Code for the paper

> van Dijck, Tjum and Fleuren, Tijn and Temizoz, Tarkan and Merzifonluoglu, Yasemin and Hendriks, Maarten and van Jaarsveld, Willem (2024). *Inventory planning in capacitated high-tech assembly systems under non-stationary demand* Available at SSRN: <https://ssrn.com/abstract=4843271>


---


![Dynaplex logo](docs/source/assets/images/logo.png)

DynaPlex is a software library for solving Markov Decision Problems and similar models (POMDP, HMM) written primarily in C++20 with bindings for python. It supports 
deep reinforcement learning, approximate dynamic programming, classical parameterized policies, and exact methods based on policy and value iteration. Models in DynaPlex are written in C++, and exposed via a generic and vectorized interface. 

DynaPlex focuses on solving problems arising in Operations Management: Supply Chain, Transportation and Logistics, Manufacturing, etc. 

---

## Citation

```bibtex
@article{van2024inventory,
  title={Inventory planning in capacitated high-tech assembly systems under non-stationary demand},
  author={van Dijck, Tjum and Fleuren, Tijn and Temizoz, Tarkan and Merzifonluoglu, Yasemin and Hendriks, Maarten and van Jaarsveld, Willem},
  journal={Available at SSRN 4843271},
  year={2024}
  url    = {https://ssrn.com/abstract=4843271}
}
```

--- 
## High-level overview of folder structure

- **`LICENSES/`**: Contains the licenses to used libraries and packages.
- **`bash/`**: Contains the files used for running on a Linux HPC.
- **`cmake/`**: Contains support functionality for building with CMake. 
- **`docs/`**: Contains the documentation.
- **`python/`**: Contains example python scripts, that can be used after building the python bindings.
- **`src/`**: Contains the main code base
  - **`executables/`**: Contains all executables you can run (you can add additional executables yourself here, that use the library).
  - **`extern/`**: Contains all external libraries used (e.g., googletest).
  - **`lib/`**: Contains all algorithms and all MDP models, you can implement your MDP in src/lib/models/models.
  - **`tests/`**: Contains all code for unit testing (supported by googletest).

---


### Prerequisites
For defining new models, you will need:
- **CMake**: Building is supported with a modern CMake version (>= 3.21), often supplied with modern C++ IDEs.

For actually training models, you have two choices. Either go all c++, in which case you need to select C++ in the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to download:
- **LibTorch**: Tests were conducted with version 2.1.0.

Alternatively, if you prefer to use python for training scripts:  
- **Python Bindings**: With pybind11 - we provide an anaconda python/environment.yml that can be used to create appropriate environment for using those bindings. 

In either case, you must provide a CMakeUserLists.txt in the root directory that provides the paths to the appropriate folders. An example (that you could copy to root where it will be ignored by git and by default invisible in some IDE) can be found in cmake/resources/CmakeUserPresets.txt

---



## Documentation

Complete documentation of DynaPlex can be found on the website: https://dynaplex-documentation.readthedocs.io

---

