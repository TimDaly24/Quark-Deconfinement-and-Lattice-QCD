# Lattice QCD — Quark Deconfinement Simulation

A Python implementation of a Lattice QCD simulation studying the confinement–deconfinement phase transition in the strong force. The simulation uses Monte Carlo methods with an SU(2) gauge group to identify the critical temperature at which quarks transition from a confined state into a quark-gluon plasma.

This was completed as a Semester 2 Computational Physics project at Maynooth University (BSc Theoretical Physics and Pure Mathematics).

---

## Overview

At extremely high temperatures ($\sim 10^9$ K), quarks gain enough energy to escape confinement and form a quark-gluon plasma. This simulation models that phase transition using:

- A 4-dimensional hypercubic lattice (3 spatial + 1 temporal dimension)
- An **SU(2) gauge group** as a simplified but physically faithful stand-in for the full SU(3) theory
- **Monte Carlo integration** via the Heatbath algorithm to generate gauge field configurations
- **Overrelaxation** to explore configuration space more efficiently
- **Tadpole improvement** to correct for lattice artefacts on coarse lattices

The two primary observables used to identify the phase transition are the **average plaquette** and the **Polyakov loop**, which together pin down the critical temperature to around $\beta = 2.2$, corresponding to $T \approx 235$ MeV.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

### Install dependencies

```bash
pip install numpy matplotlib scipy
```

---

## Structure

| File | Description |
|---|---|
| `Main_Function.py` | Core algorithm — heatbath, overrelaxation, Polyakov loop, tadpole improvement |
| `Average_Plaquette_Beta.py` | Average plaquette vs $\beta$ for a $4^3 \times 4$ lattice (reproduces Fig. 4) |
| `Thermalisation.py` | Burn-in analysis for various lattice sizes (reproduces Fig. 6) |
| `Noise_Analysis.py` | Monte Carlo noise vs lattice size — fits $\sigma = \alpha/\sqrt{V}$ (reproduces Fig. 7) |
| `Extrapolation.py` | Exponential extrapolation of lattice spacing $a$ from known $\beta$ values (reproduces Fig. 11) |
| `Jackknife_Polyakov.py` | Jackknife error analysis for the Polyakov loop and susceptibility |
| `Jackknife_Plaquette.py` | Jackknife error analysis for the average plaquette |

---

## Usage

### Run the main simulation

The core simulation runs the heatbath and overrelaxation algorithms over a range of $\beta$ values on an $8^3 \times 4$ lattice, computing the Polyakov loop expectation value and susceptibility to identify the phase transition.

```bash
python Main_Function.py
```

This produces plots of:
- $\langle P \rangle$ (Polyakov loop expectation value) vs $\beta$
- $\chi_L$ (Polyakov loop susceptibility) vs $\beta$

### Average plaquette vs $\beta$

```bash
python Average_Plaquette_Beta.py
```

Runs for $\beta \in [1.0, 3.0]$ in steps of 0.1 on a $4^3 \times 4$ lattice. Useful for validating the algorithm against known results and identifying the approximate location of the phase transition.

### Thermalisation analysis

```bash
python Thermalisation.py
```

Tracks the average plaquette over 200 sweeps for $4^3 \times 4$ and $8^3 \times 4$ lattices at $\beta = 1.9$. Confirms the lattice thermalises after approximately 20 sweeps.

---

## Key Results

The simulation identifies a clear confinement–deconfinement phase transition at:

$$\beta_c \approx 2.2 \quad \Rightarrow \quad T_c \approx 235 \text{ MeV}$$

This is consistent across both the average plaquette (kink in slope around $\beta = 2.2$) and the Polyakov loop (sharp rise from $\langle P \rangle = 0$ to $\langle P \rangle \neq 0$ at the same $\beta$). Monte Carlo noise was found to scale as $\sigma = 0.2175/\sqrt{V}$, consistent with expected lattice scaling behaviour.

---

## Lattice Parameters

The main simulation uses the following default configuration, which can be adjusted at the top of `Main_Function.py`:

```python
Nx, Ny, Nz = 8, 8, 8   # Spatial dimensions
Nt = 4                  # Temporal dimension
nwarm = 20              # Thermalisation sweeps
runs = 100              # Measurement sweeps
```

---

## Further Work

Potential extensions discussed in the paper include:

- **SU(3)**: Extending the gauge group to the full 3-colour theory
- **Dynamical fermions**: Moving beyond the quenched approximation to include quark feedback on the gauge field
- **Parallelisation**: Checkerboard masking to update independent links simultaneously
- **Larger lattices**: e.g. $16^3 \times 4$ for improved continuum limit accuracy

---

## Reference

Full project paper: *Simulating Quark Deconfinement using Lattice QCD*, Tim Daly, Maynooth University, June 2025.

