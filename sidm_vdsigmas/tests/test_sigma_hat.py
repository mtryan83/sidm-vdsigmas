# Compare outcomes to Gad-Nasr

#
# | Run | Color | $\frac{\sigma_0}{m_{dm}}$ | $w$   | $\rho_{c,LS}$ | $v_{c,LS}$ | $\rho_{c,10}$ | $v_{c,10}$ | $\frac{\sigma_{c,0}}{m_{dm}}$ | $\frac{\sigma_{c,LS}}{m_{dm}}$ | $\frac{\sigma_{c,10}}{m_{dm}}$ | $n_{c,0}$ | $n_{c,LS}$ | $n_{c,10}$ | $\hat{\sigma}_{c,0}$ | $\hat{w}_{c,0}$ |
# | --- | ----- | ---------------------- | ----- | ------------ | ------------ | ------------ | ------------ | --------------------------- | ---------------------------- | ---------------------------- | ----------- | ----------- | ----------- | ----------------- | ----------------- |
# | 1   | ——    | 5.0                    | 1e4   | 3.3          | 39.5         | 3.9          | 100.2        | 5.0                         | 4.89                         | 4.88                         | 2.8e−4      | 4.6e−4      | 3.0e−3      | 0.03              | 339               |
# | 2   | ——    | 5.5                    | 535   | 3.3          | 39.5         | 2.8          | 93.9         | 5.2                         | 4.98                         | 3.76                         | 0.09        | 0.15        | 0.58        | 0.03              | 18.2              |
# | 3   | ——    | 6                      | 298   | 3.7          | 39.8         | 1.9          | 86.3         | 5.2                         | 4.68                         | 2.74                         | 0.26        | 0.39        | 1.0         | 0.03              | 10.1              |
# | 4   | ——    | 7                      | 184.7 | 4.5          | 40.2         | 2.0          | 83.3         | 5.1                         | 4.14                         | 1.81                         | 0.53        | 0.77        | 1.52        | 0.03              | 6.3               |
# | 5   | ——    | 11                     | 103.9 | 6.0          | 40.9         | 3.9          | 85.2         | 5.1                         | 3.48                         | 0.94                         | 1.02        | 1.37        | 2.18        | 0.03              | 3.5               |
# | 6   | ——    | 42                     | 41.7  | 14.5         | 42.9         | 21.6         | 93.4         | 5.0                         | 2.13                         | 0.26                         | 1.97        | 2.4         | 3.0         | 0.03              | 1.4               |
# | 7   | ——    | 1050                   | 11.8  | 40.0         | 45.5         | 144.2        | 104.5        | 5.1                         | 1.15                         | 0.07                         | 3.01        | 3.3         | 3.5         | 0.03              | 0.4               |
# | 8   | ——    | 5e6                    | 1.0   | 92.3         | 47.7         | 666.7        | 115.3        | 5.0                         | 0.71                         | 0.03                         | 3.67        | 3.7         | 3.8         | 0.03              | 0.03              |
# | 9   | ——    | 90                     | 1e4   | 0.016        | 31.4         | 0.016        | 77.4         | 90.0                        | 88                           | 87.9                         | 2.8e-4      | 2.9e-4      | 1.8e-3      | 0.5               | 339               |
# | 10  | ——    | 96                     | 535   | 0.016        | 31.4         | 0.011        | 72.6         | 91.6                        | 89.4                         | 74.5                         | 0.09        | 0.1         | 0.4         | 0.5               | 18.2              |
# | 11  | ——    | 105                    | 298   | 0.017        | 31.5         | 0.009        | 68.8         | 91.2                        | 88.5                         | 59.3                         | 0.26        | 0.27        | 0.82        | 0.5               | 10.1              |
# | 12  | ——    | 125.4                  | 184.7 | 0.017        | 31.5         | 0.009        | 67.2         | 90.8                        | 87.2                         | 44.0                         | 0.53        | 0.56        | 1.29        | 0.5               | 6.3               |
# | 13  | ——    | 195                    | 103.9 | 0.017        | 31.6         | 0.01         | 65.6         | 91.1                        | 84.7                         | 28.5                         | 1.02        | 1.09        | 1.9         | 0.5               | 3.5               |
# | 14  | ——    | 770                    | 41.7  | 0.021        | 31.9         | 0.02         | 67.3         | 92.3                        | 76.2                         | 12.1                         | 1.97        | 2.1         | 2.8         | 0.5               | 1.4               |
# | 15  | ——    | 1.9e4                  | 11.8  | 0.029        | 32.6         | 0.09         | 73.8         | 92.0                        | 60.8                         | 4.12                         | 3.01        | 3.1         | 3.4         | 0.5               | 0.4               |
# | 16  | ——    | 9e7                    | 1.0   | 0.041        | 33.2         | 0.44         | 83.3         | 90.8                        | 48.8                         | 1.58                         | 3.67        | 3.7         | 3.8         | 0.5               | 0.03              |
#
# $\rho_{LS}$ units are $10^{10}\,\frac{{\rm M}_{\odot}}{{\rm kpc}^3}$
# $\rho_{10}$ units are *$10^{14}\,\frac{{\rm M}_{\odot}}{{\rm kpc}^3}$*
# $w$ and $v$ units are $\frac{\rm km}{\rm s}$
# $\sigma$ units are $\frac{{\rm cm}^2}{\rm g}$
#

from io import StringIO

import numpy as np
import pandas as pd
from IPython.display import display
from unyt import (
    gravitational_constant as G0,
)
from unyt import (
    kilometer as km,
)
from unyt import (
    kiloparsec as kpc,
)
from unyt import (
    second,
    unyt_array,
    unyt_quantity,
)
from unyt import (
    solar_mass as Msun,
)

from sidm_vdsigmas import Moller, sigunit
from sidm_vdsigmas.interaction import Interaction

Msun.convert_to_units("solar_mass")


def get_outcomes():
    raw = """
Run Color sigconst w rhoLS vLS rho10 v10 sig0  sigLS sig10 n0 nLS n10 sighat0 what0
1  —— 	5.0 	1e4 	3.3 	39.5 	3.9 	100.2 	5.0 	4.89 	4.88 	2.8e-4 	4.6e-4 	3.0e-3 	0.03 	339
2  —— 	5.5 	535 	3.3 	39.5 	2.8 	93.9 	5.2 	4.98 	3.76 	0.09 	0.15 	0.58 	0.03 	18.2
3  —— 	6 	298 	3.7 	39.8 	1.9 	86.3 	5.2 	4.68 	2.74 	0.26 	0.39 	1.0 	0.03 	10.1
4  —— 	7 	184.7 	4.5 	40.2 	2.0 	83.3 	5.1 	4.14 	1.81 	0.53 	0.77 	1.52 	0.03 	6.3
5  —— 	11 	103.9 	6.0 	40.9 	3.9 	85.2 	5.1 	3.48 	0.94 	1.02 	1.37 	2.18 	0.03 	3.5
6  —— 	42 	41.7 	14.5 	42.9 	21.6 	93.4 	5.0 	2.13 	0.26 	1.97 	2.4 	3.0 	0.03 	1.4
7  —— 	1050 	11.8 	40.0 	45.5 	144.2 	104.5 	5.1 	1.15 	0.07 	3.01 	3.3 	3.5 	0.03 	0.4
8  —— 	5e6 	1.0 	92.3 	47.7 	666.7 	115.3 	5.0 	0.71 	0.03 	3.67 	3.7 	3.8 	0.03 	0.03
9  —— 	90 	1e4 	0.016 	31.4 	0.016 	77.4 	90.0 	88 	87.9 	2.8e-4 	2.9e-4 	1.8e-3 	0.5 	339
10  —— 	96 	535 	0.016 	31.4 	0.011 	72.6 	91.6 	89.4 	74.5 	0.09 	0.1 	0.4 	0.5 	18.2
11  —— 	105 	298 	0.017 	31.5 	0.009 	68.8 	91.2 	88.5 	59.3 	0.26 	0.27 	0.82 	0.5 	10.1
12  —— 	125.4 	184.7 	0.017 	31.5 	0.009 	67.2 	90.8 	87.2 	44.0 	0.53 	0.56 	1.29 	0.5 	6.3
13  —— 	195 	103.9 	0.017 	31.6 	0.01 	65.6 	91.1 	84.7 	28.5 	1.02 	1.09 	1.9 	0.5 	3.5
14  —— 	770 	41.7 	0.021 	31.9 	0.02 	67.3 	92.3 	76.2 	12.1 	1.97 	2.1 	2.8 	0.5 	1.4
15  —— 	1.9e4 	11.8 	0.029 	32.6 	0.09 	73.8 	92.0 	60.8 	4.12 	3.01 	3.1 	3.4 	0.5 	0.4
16  —— 	9e7 	1.0 	0.041 	33.2 	0.44 	83.3 	90.8 	48.8 	1.58 	3.67 	3.7 	3.8 	0.5 	0.03
"""
    outcomes = pd.read_table(StringIO(raw), sep=r"\s+")
    outcomes = outcomes.apply(pd.to_numeric, args=("coerce",))
    outcomes["rhoLS"] = outcomes["rhoLS"] * 1e10
    outcomes["rho10"] = outcomes["rho10"] * 1e14
    return outcomes


outcomes = get_outcomes()


def get_LS10_params(
    rho_s: unyt_quantity,
    v_max: unyt_quantity,
    *,
    sigma: Interaction,
    alpha=2.2,
    xi=None,
):
    """
    Follow recipe from Gad-Nasr 23
    sigma should be an Interaction object
    """
    if xi is None:
        xi = 0.11

    # maximal core quantities
    rho0 = 2.4 * rho_s
    v0 = 0.64 * v_max
    # for reference, r_c,0 = 0.19 r_s, if we assume v_c,0 = 0.64*vmax and rho_c,0=2.4*rho_s, and NFW
    # using vcirc^2(r) = 4*π*G*ρs*rs^3*f(r)/r, f(r)=ln(1+r/rs)+(r/rs)/(1+(r/rs))
    # Computed using the defs from Eq 9 in Outmezguine23
    # possible factor of 3 missing?
    r0 = (v0 / np.sqrt(4 * np.pi * G0 * rho0)).to("kpc")
    M0 = v0**3 * rho0 / np.sqrt(4 * np.pi * G0**3 * rho0**3)
    what0 = sigma.v0 / v0
    sighat0 = sigma.sigma_hat_fun(1 / what0, vn=v0, rhon=rho0)
    sigma0om = sigma.sigconst
    n0 = sigma.n(v0 / sigma.v0, use_K5=True)
    delta = 1 - n0 + alpha / (alpha - 2)

    # LS transition quantities
    vLSv0 = sighat0 ** (-1 / delta)
    vLS = vLSv0 * v0
    nLS = sigma.n(vLS / sigma.v0, use_K5=False)
    rhoLSrho0 = vLSv0 ** ((2 * alpha) / (alpha - 2))
    rhoLS = rhoLSrho0 * rho0
    MLSM0 = vLSv0 ** ((6 - 2 * alpha) / (2 - alpha))
    MLS = MLSM0 * M0
    rLSr0 = vLSv0 ** (-2 / (alpha - 2))
    rLS = rLSr0 * r0
    sighatLS = sigma.sigma_hat_fun(vLS / sigma.v0, vn=vLS, rhon=rhoLS)

    # gamma=10 quantities
    v10vLS = np.exp((0.74 + 0.008 * nLS**2.5) / (nLS**0.03))
    rho10rhoLS = np.exp((8.43 + 0.18 * nLS**2.21) / (nLS**0.013))

    rho10 = rho10rhoLS * rhoLS
    v10 = v10vLS * vLS
    n10 = sigma.n(v10 / sigma.v0, use_K5=False)
    beta = (10 - xi * (1 + 7 * n10)) / 28

    M10 = (
        np.sqrt(6 / np.pi) * ((1 + xi) / (G0)) ** (3 / 2) * (v10) ** 3 / np.sqrt(rho10)
    )
    M10 = M10.to("Msun")

    # It looks like sigma10 is defined as sigma0om * K3 in Outmezguine
    # Assume it should be sigma0om * K5 according to the note in GN
    # But we should probably use Keff in the SMFP regime
    sig10 = sigma.sigconst * sigma.Keff(v10 / sigma.v0)
    sigLS = sigma.sigconst * sigma.Keff(vLS / sigma.v0)
    sig0 = sigma.sigconst * sigma.K5(v0 / sigma.v0)
    output = {
        "v0": v0,
        "rho0": rho0,
        "vmax": v_max,
        "rho_s": rho_s,
        "M0": M0,
        "r0": r0,
        "what0": what0,
        "sighat0": sighat0,
        "sigconst": sigma0om,
        "n0": n0,
        "sig0": sig0,
        "vLS": vLS,
        "nLS": nLS,
        "rhoLS": rhoLS,
        "sigLS": sigLS,
        "sighatLS": sighatLS,
        "MLS": MLS,
        "rLS": rLS,
        "v10": v10,
        "n10": n10,
        "rho10": rho10,
        "sig10": sig10,
        "M10": M10,
        "vLSv0": vLSv0,
        "rhoLSrho0": rhoLSrho0,
        "v10vLS": v10vLS,
        "rho10rhoLS": rho10rhoLS,
        "MLSM0": MLSM0,
        "rLSr0": rLSr0,
        "beta": beta,
        "xi": xi,
        "delta": delta,
    }

    return output


def compare_outcome(
    out: pd.DataFrame,
    *,
    rho_s=(2e7 * Msun / kpc**3),
    rs=3 * kpc,
    Vmax=45.9 * km / second,
    alphaX=0.01,
    # sigma_fun = None,
    # p1p2fun = None,
    sigma=None,
    sigma_params=None,
    verbose=False,
):
    # out is a single row from outcomes
    if sigma_params is None:
        sigma_params = {"potential_kind": "repulsive", "potential_ave": "V"}
    w = out["w"] * km / second
    sigcons = out["sigconst"] * sigunit
    if w is None:
        w = 298 * km / second
    if sigcons is None:
        sigcons = 6 * sigunit
    if verbose:
        print(f"{w=:.4} {sigcons=:.4}")
    if sigma is None:
        sigma = Moller(sigconst=sigcons, w=w, alphaX=0.01)

    output = get_LS10_params(rho_s=rho_s, v_max=Vmax, sigma=sigma)
    output["w"] = (w).to("km/s")

    labels = out.index
    perdiffs = np.zeros(len(labels))
    outtab = np.zeros_like(perdiffs)
    for i, lab in enumerate(labels):
        if lab not in output:
            continue
        ol = output[lab]
        if isinstance(ol, (unyt_array, unyt_quantity)):
            ol = ol.v
        outtab[i] = ol
        try:
            perdiff = np.abs(out[lab] - ol) / out[lab]
        except KeyError:
            if verbose:
                print(f"Failed to process {lab}")
            perdiffs[i] = -1
            continue
        if verbose:
            print(f"{lab}:\t{out[lab]:.4}\t{output[lab]:.4}\t%diff: {perdiff:.1%}")
        perdiffs[i] = perdiff
    return perdiffs, outtab


compare_outcome(outcomes.iloc[2, :], verbose=True)

outcomes = get_outcomes()
nrows, ncols = outcomes.shape
perdiffs = np.zeros((nrows, ncols))
outtab = np.zeros_like(perdiffs)

for i in range(nrows):
    perdiff, ot = compare_outcome(
        outcomes.iloc[i, :],
    )
    perdiffs[i, :] = np.round(perdiff * 100, 2)
    outtab[i, :] = ot
perdiffs = pd.DataFrame(perdiffs, columns=outcomes.columns)
outtab = pd.DataFrame(outtab, columns=outcomes.columns)
outtab["rho10"] /= 1e14
outtab["rhoLS"] /= 1e10
display(perdiffs.iloc[:, 2:])
display(outtab.iloc[:, 2:])
if (perdiffs > 100).any().any():
    raise Exception("Autotesting FAILED!")  # noqa
