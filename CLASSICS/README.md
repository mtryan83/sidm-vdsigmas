# CLASSICS
**CalcuLAtionS of Self Interaction Cross Sections**

This code provides approximate analytial expressions for the self-scattering cross sections of dark matter particles interacting via a Yukawa potential both in the semi-classical regime (k / mphi > 1, where k is the dark matter momentum and mphi is the mediator mass) and in the quantum regime (1 > k / mphi). One can calculate the momentum transfer cross section and the viscosity cross section for distinguishable particles, and the spin-averaged viscosity cross section for identical particles. To obtain the velocity-averaged cross sections for a Maxwell-Boltzmann distribution, the code interpolates tabulated numerical results. 

For details, see B. Colquhoun, S. Heeba, F. Kahlhoefer, L. Sagunski and S. Tulin, *The Semi-Classical Regime for Dark Matter Self-Interactions*, [arXiv:2011.04679](https://arxiv.org/abs/2011.04679). 

For the Hulthen approximation, used to calculate the S-wave scattering cross sections in the quantum regime, see S. Tulin, H. Yu and K. Zurek, *Beyond Collisionless Dark Matter: Particle Physics Dynamics for Dark Matter Halo Structure*, [arXiv:1302.3898](https://arxiv.org/abs/1302.3898).

For example usage, see *cross_sections.py*.

The tabulated values are provided in the following format (following the notation from [arXiv:2011.04679](https://arxiv.org/abs/2011.04679)):
* Column 1: beta_0
* Column 2: kappa_0
* Column 3: m_phi^2/pi * sigma_X^Y, where
  * X = T or V for distinguishable particles and X = even, odd, scalar, fermion or vector for identical particles
  * Y = attractive, repulsive


