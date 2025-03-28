# ECT-QST

Enhanced Compressive Threshold Quantum State Tomography (ECT-QST) is a procedure enabling quantum state tomography of systems of qudits while optimizing the number of **measurement settings** to be used. It is based on ideas from [threshold quantum state tomography (tQST)](https://doi.org/10.1063/5.0219143), which is a procedure that performs quantum state tomography optimizing the number of **projective measurements**.

A preprint detailing ECT-QST approach has been [published in Physical Review A](https://link.aps.org/doi/10.1103/PhysRevA.111.032436).
A preprint is available on [arXiv](https://arxiv.org/abs/2502.10031).

The main files of the distribution are imported as
```
import density_matrix_tool as dmt
import ect_tomography_qudit as ect
import maximum_likelihood as ml
```
 * ``dmt`` contains general routines to deal with matrices (_e.g._, calculation of fidelities, generation of certain type of matrices)
 * ``ect`` contains routines the are related to ECT-QST (_e.g._, computing the measurement settings that will be measured)
 * ``ml`` contains routines that perform maximum likelihood reconstruction of the density matrix given the measurements.

Analogously to tQST, ECT-QST proceeds by the following steps:

## Instantiation of the ``ECTtomography`` class.

One needs to specify the dimension $d$ of the qudits and their number $N$.
```
d = 3
N = 2
tomo = ect.ECTtomography(d,N, verbose=True)
```
where the ``verbose`` flag results in informative output on the various steps.

## Measurement of the diagonal of the the density operator

The first step is to perform a measurement of the system in the computational basis.   
For $d=2$ this is the setting that measures the $\sigma_z$ [Pauli matrix](https://en.wikipedia.org/wiki/Pauli_matrices) on all the qubits. In the general case, the computational basis is defined here as the set of common eigenstates of the maximum abelian subspace of $SU(d)$ operators on each qubit.   
For $d=3$, the computational basis is defined as the common eigenstates of the $\lambda_3$ and $\lambda_8$ [Gell-Mann matrices](https://en.wikipedia.org/wiki/Gell-Mann_matrices).

In any case, the vectors representing the states that measure the diagonal element of the density matrix are the rows of the $(d^N, d^N)$ identity matrix. Once the corresponding measurement values are placed is a numpy array, let's call it ``diagonal``, one needs to decide the value of a threshold ``t``, that indicates which are the states of the computational basis the provide significant information.    
If ``t = np.min(diagonal)``, then complete quantum state tomography will be performed.

The choice of the threshold is crucial: ideally, it is larger than the values of the diagonal that correspond to "noise" and smaller than the values of the diagonal elements that correspond to "signal". In case of doubt, ``dmt.gini_index(diagonal)`` will return an estimate of the threshold based on the [Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient).

### Important notice

The elements of ``diagonal`` need not to be normalized to one, but the threshold ``t`` and all subsequent counts **must** be given with the same normalization.

## Finding the settings to be measured

The ECT tomography algorithm is based on finding the settings whose projectors give the largest information on the real and imaginary part of the matrix elements to be further determined. These settings come with a weight that estimates the importance of their contribution.   
These are found by calling
```
settings, weights = tomo.ECT_settings(diagonal, t)
```

On outputs, `weight` is an numpy array with the weights, and ``settings`` is a list of $N$ strings, denoting which setting to measure on the various qubit. Settings are identified using the following notation
* `Z` denotes any element of the maximum abelian subspace. For a qubit, this would be $\sigma_z$. For a qutrit, either $\lambda_3$ or $\lambda_8$.
* `Rn.m` is the $d \times d$ matrix which has all zeros, except a single $1$ in position $(n,m)$ and $(m,n)$. For a qubit, the only possibility is `R0.1` which is $\sigma_x$. For a qutrit there are three possibilities; for example ``R0.2`` is the $\lambda_4$  [Gell-Mann matrix](https://en.wikipedia.org/wiki/Gell-Mann_matrices).
* `In.m` is the $d \times d$ matrix which has all zeros, except a single $i$ in position $(n,m)$ and a corresponding $-i$ in position $(m,n)$. For a qudit, this is $\sigma_y$. For a qutrit there are again three possibilities; for example ``I1.2`` is the $\lambda_7$ [Gell-Mann matrix](https://en.wikipedia.org/wiki/Gell-Mann_matrices).

The list of ``settings`` is ordered by decreasing weight. It is therefore recommended to begin measurements from the first onward.

### Measuring a setting

Each setting provides $d^N$ projector operators, each of which need to be further measured.
The method ``tomo.projector_names_of_setting(s)`` provides a list of identifiers. Each identifier is, in turn, a list of names of single-qudit projector, with the following notation:

* `Zn.n`, the $n$th computational basis state, that is a $d$-dimensional vector that is all zeros except $1$ at position $n$.
* `Xn.m` or `xn.m`. These are the eigenvectors with eingevalues $1$ and $-1$ of ``Rn.m``, respectively.
* `Yn.m` or `yn.m`. These are the eigenvectors with eingevalues $1$ and $-1$ of ``In.m``, respectively.

The actual projector in $d^N$ is the Korenecker product of these single-qudit projectors, and can be obtained by ``tomo.projector_from_name(name)``.

In the $d=2$ case the method prints also the projector names with the usual notation, if the class is instantiated with ``verbose=True``.

## Performing the maximum likelihood reconstruction

Maximum likelihood reconstruction is performed by a class instantiated as
```
maxlik = ml.Maximum_likelihood_tomography([d]*N)
```

One needs to pass to it the states measured $|\phi_k\rangle$ and the corresponding counts $c_k = \langle \phi_k | \rho | \phi_k \rangle$.

States are passed as a $(M, d^N)$ numpy array ``V``, where $M$ is the number of measurements. Hence, the states are the rows of the array $V$. The corresponding counts are passed as a $M$-dimensional numpy array ``C``.

This is done using the method ``maxlik.set_counts(V,C)``.

After this, one can optimize the likelihood by calling ``maxlik.minimize_pool_mt(T)``. This performs ``T`` local optimizations in parallel. The reconstructed density matrix can then be obtained by
```
reconstructed_rho = maxlik.model_density_matrix()
```

There are several models that one can use for the density matrix. The most unbiased is to write it as $\rho = T T^\dagger$, where $T$ is a triangular matrix. Given the large number of parameters of $T$ (=$d^N$), this optimization is by far the most time consuming, but should be reasonably fast for small values of $d$ and $N$.

Another model writes $\rho = M M^\dagger$, where $M$ is a $(d^N, m)$ matrix (with $m \ll d^N$). The rationale is that $m$ is of the order of the rank expected for $\rho$. The validity of this approximation can be checked by verifying that $\mathrm{tr}(\rho^2) m \gg 1$.    
This model is requested by instantiating
```
maxlik = ml.Maximum_likelihood_tomography([d]*N, model=ml.model_g5, model_params=M)
```

### Projectors and counts

Let us assume that one has a list ``L`` of projectors names that have been measured (this list **must** include the computational basis, that is the initial diagonal measurement), and a list ``c`` of the corresponding counts.
The arrays ``V`` and ``C`` can be generated as
```
V = np.array([tomo.projector_of_name(x) for x in L])
C = np.array(c)
```

## Recovery of the original tQST method

The ``ECTtomography`` class contains a method to recover projectors implementing the original [tQST approach](https://doi.org/10.1063/5.0219143).
Although these projectors do not turn out always to be the same, they nevertheless enable a reconstruction of the density matrix with the same quality.

The projectors are obtained using the method 
```
projs, names = tomo.tQST_projectors(diagonal, t)
```
where ``projs`` are $d^N$-dimensional numpy arrays. The names of the projectors follow the same notation described before.

In the $d=2$ case the method prints also the projector names with the usual notation, if the class is instantiated with ``verbose=True``.
