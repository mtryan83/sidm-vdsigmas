# Intro

This is a basic README for the sidm-vdsigmas package.

# Installation

This may eventually end on PyPi or conda-forge, but for now, you'll 
need to clone this repository:
```sh
git clone https://github.com/mtryan83/sidm-vdsigmas.git
```
or (if you have ssh keys set up)
```sh
git clone git@github.com:mtryan83/sidm-vdsigmas.git
```
and then install locally:
```sh
cd sidm-vdsigmas
python -m pip install -e .
```

Note that we've installed the package as an editable (the `-e` flag)
in case you want to add your own cross sections or other functions.

## The CLASSICS package
One of the provided cross sections is based on the CLASSICS package, with a
repo [here](https://github.com/kahlhoefer/CLASSICS). Our intention is use this
as a dependency, however, CLASSICS does not currently provide any means to 
install it (like a `pyproject.toml` or `setup.py`). Instead the expected use
appears to be to copy the entire folder into your current working directory and
go from there. This is what we will do **temporarily** until the installation
mechanism is updated. And by that, I mean include the entire CLASSICS folder
as a subdirectory of the `sidm-vdsigmas` package.

# Examples

```python
from sidm-vdsigmas import Rutherford, CLASSICS, sigunit
from unyt import kilometer as km, second, unyt_quantity

# Compute effective constant cross section from 2205.03392
vmax = 30 * km/second # generic dwarf galaxy
sigma = Rutherford(sigconst=147 * sigunit, w=40 * km/second)

print(f"effective constant cross section: {sigma.eff(vmax):.2g}")
# effective constant cross section: 41.96 cm**2/g

# use particle physics quantities
sigma = Rutherford(
    m=unyt_quantity(9.7,"GeV/c**2"),
    mphi=unyt_quantity(32,"keV/c**2"),
    alphaX=1e-6
)
# Note that we use a slightly different definition of K5 to the one in 
# 2205.03392. To reproduce their result, we need to modify _x_s_scaling
sigma.x_s_scaling = 1
vmax = 7.97 * km/second
print(f"effective constant cross section: {sigma.eff(vmax):.2g}")
# effective constant cross section: 7.1 cm**2/g


# compute sigma_hat from 2312.09296
sigma = Moller(sigconst=6 * sigunit, w=298 * km/second)

# note that internally, w = m_phi/m_chi (i.e. c=1), so to normalize by a 
# dimensionful w, use sigma.v0
what = 29.5 * km/second / sigma.v0
vn = 45.9 * km/second # using vmax as the normalization factor
rhon = unyt_quantity(2e7, "Msun/kpc**3") # using rhos as the normalization
print(f"sigma_hat: {sigma.sigma_hat_fun(what, vn=vn, rhon=rhon)}")
# sigma_hat: 0.03
```
