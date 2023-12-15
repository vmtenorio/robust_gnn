# Robust Graph Neural Network based on Graph Denoising

Repository for the paper Robust Graph Neural Network based on Graph Denoising, by Victor M. Tenorio, Samuel Rey and Antonio G. Marques from King Juan Carlos University, in Madrid, Spain. The abstract of the paper reads as follows:

> Graph Neural Networks (GNNs) have emerged as a notorious alternative to address learning problems dealing with non-Euclidean datasets. However, although most works assume that the graph is perfectly known, the observed topology is prone to errors stemming from observational noise, graph-learning limitations, or adversarial attacks. If ignored, these perturbations may drastically hinder the performance of GNNs. To address this limitation, this work proposes a robust implementation of GNNs that explicitly accounts for the presence of perturbations in the observed topology. For any task involving GNNs, our core idea is to i) solve an optimization problem not only over the learnable parameters of the GNN but also over the true graph, and ii) augment the fitting cost with a term accounting for discrepancies on the graph. Specifically, we consider a convolutional GNN based on graph filters and follow an alternating optimization approach to handle the (non-differentiable and constrained) optimization problem by combining gradient descent and projected proximal updates. The resulting algorithm is not limited to a particular type of graph and is amenable to incorporating prior information about the perturbations. Finally, we assess the performance of the proposed method through several numerical experiments.

The paper was submitted and presented in the 2023 Asilomar Conference on Signals, Systems, and Computers (Oct. 29th - Nov 1st, 2023).

## Citing

To cite the following paper please use

```
@inproceedings{tenorio23robustgnn,
  author={Tenorio, Victor M. and Rey, Samuel and Marques, Antonio G.},
  booktitle={Asilomar Conference on Signals, Systems, and Computers}, 
  title={Robust Graph Neural Network based on Graph Denoising}, 
  month={Oct. 29th - Nov 1st},
  year={2023}
}
```
