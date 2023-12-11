# GFN1-xTB

## Introduction

A semi-empirical DFT model based on a tight-binding approximative parametrization including dispersive energy of organic molecules. This method was developed by Grimme's group [1,2].

This code was developed to the Remote Programing Project module of the EMTCCM masters program (course 2023/24) by Felipe Reibnitz Willemann.

All guidelines, theoretical revision, test coordinate files, debugging files and parameters file were provided by Professor Jeremy Harvey of KU Leuven.

## Usage

_ATENTION: Requires Python 3.9.18 and Numpy 1.26.0._

1. Prepare any organic molecule composed of the elements H, C, N and O in the XYZ format (strictly) and save them in a separate directory (preferably).

2. Run the file `src/main.py` from any working directory in a *Python* environment with *Numpy* installed and follow instructions.

## References

[1] “Extended tight-binding quantum chemistry methods”, C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher and S. Grimme, WIREs Comput. Mol. Sci. 2021, 11 e1493, DOI: <https://doi.org/10.1002/wcms.1493>.

[2] “A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parametrized for All spd-Block Elements (Z = 1-86)”, S. Grimme, C. Bannwarth and P. Shushkov, J. Chem. Theory Comput. 2017, 13, 1989-2009, <https://doi.org/10.1021/acs.jctc.7b00118>.