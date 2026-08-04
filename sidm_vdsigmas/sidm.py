from typing import cast

import numpy as np
from unyt.array import unyt_array, unyt_quantity

"""
Module for the SIDM class and related functionality
"""


class SIDM:
    """Class to store SIDM parameters

    This class maintains a self-consistent set of SIDM parameters, including
    SIDM particle mass (mX), SIDM mediator mass (mphi), SIDM fine structure
    constant (alphaX), and mass ratio (w=mX/mphi). Note that only one of
    (mphi,w) needs to be specified. The other will be auto-calculated.

    Inputs:
        mX: unyt_quantity
        Mass of the SIDM particle

        mphi: unyt_quantity, optional
        Mass of the SIDM mediator. If not provided, will by computed based
        on mX and w.

        w: float | unyt_quantity, optional
        Mass ratio of mphi to mX. If not provided will be computed based on
        mX and w.

        alphaX: float | unyt_quantity
        SIDM fine structure constant

    Raises:
        ValueError if one or more of mX, alphaX, or (mphi/w) are missing or
        are inconsistent.
    """

    mX: unyt_quantity
    mphi: unyt_quantity
    alphaX: float
    w: float

    def __init__(
        self,
        *,
        mX: unyt_quantity | None = None,
        mphi: unyt_quantity | None = None,
        alphaX: float | None = None,
        w: float | unyt_quantity | None = None,
    ):
        self.alphaX = 0.01 if alphaX is None else alphaX
        if sum(x is None for x in [mX, mphi, w]) > 1:
            raise ValueError("Must provide two of w, mphi, mX")
        w = w.v if isinstance(w, unyt_quantity | unyt_array) else w
        if w is None:
            assert mX is not None
            assert mphi is not None
            self.mX = mX
            self.mphi = mphi
            self.w = cast(unyt_quantity, mphi / mX).v
        elif mX is None:
            assert mphi is not None
            assert w is not None
            self.mphi = mphi
            self.w = w
            self.mX = unyt_quantity(mphi / w)
        elif mphi is None:
            assert w is not None
            assert mX is not None
            self.mX = mX
            self.w = w
            self.mphi = unyt_quantity(mX * w)
        else:
            self.mX = mX
            self.mphi = mphi
            self.w = w
            if not np.isclose(unyt_quantity(self.mphi / self.mX).v, w):
                raise ValueError(
                    f"{w=} and mphi/mX={self.mphi / self.mX} are inconsistent!"
                )

    def __repr__(self):
        phiunit = "GeV/c**2" if np.log10(self.mphi.to("MeV/c**2")) > 1 else "MeV/c**2"
        return f"SIDM(mχ={self.mX:.4},mϕ={self.mphi.to(phiunit):.4},w={self.w:.4},α={self.alphaX})"
