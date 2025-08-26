import numpy as np

from sidm_vdsigmas.interaction import Interaction

"""
This module implements the approximate cross section currently (2025, c70c760)
currently provided in the GIZMO multi-physics code.
"""

class Gizmo(Interaction):
	"""Gizmo cross section

	The cross section is based on the self-interaction kernel defined in the 
    sidm/sidm_core.c file as of commit c70c760. This is a rough approximation
    of a Moller/Rutherford-type viscosity cross section.
		
	"""
	
    name = 'Gizmo'
    file_name = name
    def __call__(self,v):
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
        return sigma0 * self.hat(v/w)

    def hat(self,x):
	    r"""Compute the value of the dimensionless component of the cross section at the specified dimensionless velocity

        Inputs:
            x: float | unyt_like(dimensionless)
            Velocity to calculate at

        Returns:
            unyt_like(dimensionless)
            Value of the cross section at the specified velocity. Will have same shape as v
        """
        x = np.maximum(x,1/50)
        return 1./(1. + x**4.)
