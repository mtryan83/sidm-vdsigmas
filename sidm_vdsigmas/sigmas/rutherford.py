import numpy as np

from sidm_vdsigmas.interaction import Interaction

"""
This module implements Rutherford cross sections. Currently only the
viscosity cross section is supported.
"""


class Rutherford(Interaction):
    """Rutherford viscosity cross section

    The cross section is computed from the analytic dimensionless cross section
    defined in Yang 2023 (arXiv:2305.05067)

    """

    name = "RutherfordV"
    file_name = name

    def __call__(self, v):
        """Compute the value of the cross section at the specified velocity

        Note that this is just a scaled version of hat(x).

        Inputs:
            v: unyt_like
            Velocity to calculate at

        Returns:
            unyt_like
            Value of the cross section at the specified velocity. Will have same shape as v

        See Also:
            hat: dimensionless cross section
        """
        w = self.v0
        sigma0 = self.sigconst
        return sigma0 * self.hat(v / w)

    def hat(self, x):
        r"""Compute the value of the dimensionless component of the cross section at the specified dimensionless velocity

        Inputs:
            x: float | unyt_like(dimensionless)
            Velocity to calculate at

        Returns:
            unyt_like(dimensionless)
            Value of the cross section at the specified velocity. Will have same shape as v
        """
        x = np.maximum(x, 1 / 50)
        return 6 * x**-6 * (-2 * x**2 + (2 + x**2) * np.log(1 + x**2))
