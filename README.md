# Distance Graph Fusion (DGF) and Similarity Graph Fusion (SGF)

This repository contains the MATLAB code for DGF and SGF introduced in the following paper 

	Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering (ICDM 2019) 


### Preparation
* Windows 64bit
Add some helper files to MATLAB path by `addpath('MinMaxSelection')` command in MATLAB command window.
* Windows 32bit and Mac OS
Add some helper files to MATLAB path by `addpath('MinMaxSelection')` command in MATLAB command window. Then recompile the helper functions by running `minmax_install`.


### Example usage
```MATLAB
load('data\handwritten.mat');
knn=15; beta=1e-6; gamma=1e1;
[nmi, label] = DGF(X, Y, knn, beta, gamma);
```


### Citation
If you find this algorithm useful in your research, please consider citing:

	@inproceedings{liang2019consistency,
	  title={Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering},
	  author={Youwei Liang, Dong Huang, and Chang-Dong Wang},
	  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
	  year={2019}
	}
