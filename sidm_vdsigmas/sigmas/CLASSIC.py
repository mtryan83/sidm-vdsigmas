
from unyt import speed_of_light as c0

import cross_sections as classics

from sidm_vdsigmas.interaction import Interaction

"""
This module implements CLASSIC style cross sections. 
"""

class CLASSIC(Interaction):
    r"""Class for computing CLASSIC-style cross section
    
    This class serves as a wrapper for the cross sections computed using the
    methods in Colquhon 2021 (arXiv:2011.04679). 
    
    Inputs:
        m,mphi,alphaX,sigconst,w,sidm:
        See definitions in interaction.Interaction
        
        mode: str, optional
        Passed to the CLASSICS module. Can take one of the following values: 
         'T': Returns the momentum transfer cross section for distinguishable particles
         'V': Returns the viscosity cross section for distinguishable particles
         'even': Returns the viscosity cross section for identical particles with even spatial wave function
         'odd': Returns the viscosity cross section for identical particles with odd spatial wave function
         'scalar: Returns the viscosity cross section for identical scalar particles
         'fermion': Returns the viscosity cross section for identical fermions (averaged over initial spins)
         'vector': Returns the viscosity cross section for identical vector particles (averaged over initial spins)
          If no mode is specified, the default option is 'T' (configurable based on CLASSIC.mode)
          
        sign: str, optional
        Passed to the CLASSICs module. Can be either attractive or repulsive/
        Default is attractive (configurable based on CLASSIC.sign)
        
    Attributes:
        iden: float
        1/2 if particles are identical ('T' or 'V' modes), 1 otherwise
    """

    def __init__(self,*args,mode=None,sign=None,**kwargs):
        super().__init__(*args,**kwargs)
        if mode is None:
            mode = 'T' if CLASSIC.mode is None else CLASSIC.mode
        if sign is None:
            sign = 'attractive' if CLASSIC.sign is None else CLASSIC.sign
        if self.alphaX is None:
            raise ValueError('CLASSICs cross sections must have alphaX defined.')
        self.mode = mode
        self.sign = sign
        self.iden = 1/2 if mode in ('T','V') else 1 # 1/2 for identical particles
        # CLASSICs cross-sections have an additional non-dimensional geometric factor that needs to be
        # included when figuring out sigconst
        sig0 = self._sigma_nd(self.v0 / 50)
        self.sigscale = self.sigconst / sig0

    @property
    def name(self):
        sign = '+' if self.sign == 'attractive' else '-'
        return f'CLASSIC ({sign},{self.mode})'

    @property
    def file_name(self):
        sign = '+' if self.sign == 'attractive' else '-'
        return f'CLASSIC{sign}{self.mode}'

    @classmethod
    def gen_name(cls):
        """Return the current CLASSIC type assigned to the class
        """
        sign = '+' if cls.sign == 'attractive' else '-'
        return f'CLASSIC{sign}{cls.mode}'
        
    def __call__(self,v):
        return self.sigscale * self.sigma_nd(v)
        """Compute the value of the cross section at the specified velocity

        Note that this should just be a scaled version of hat(x), however
        a slight difference (though mathematically equivalent) in how the
        scalings are defined may cause small numerical differences.

        Inputs:
            v: unyt_like
            Velocity to calculate at

        Returns:
            unyt_like
            Value of the cross section at the specified velocity. Will have same shape as v

        See Also:
            hat: dimensionless cross section
        """
        return self.sigscale * self._sigma_nd(v)

    def hat(self,x):
        r"""Compute the value of the dimensionless component of the cross section at the specified dimensionless velocity

        Inputs:
            x: float | unyt_like(dimensionless)
            Velocity to calculate at

        Returns:
            unyt_like(dimensionless)
            Value of the cross section at the specified velocity. Will have same shape as v
        """
        kappa = x/2
        beta = 2*self.alphaX/self.w / x**2
        return self.iden*classics.sigma_combined(kappa, beta, mode=self.mode, sign=self.sign)

    def _sigma_nd(self,v):
        """Compute dimensionless sigma directly from velocity instead of scaled velocity

        This should produce equivalent results to calling self.hat(v/self.v0),
        but since beta is defined here as w*c^2/v^2 instead of 1/(w*(v/v0)^2)
        there may be slight differences.
        
        """
        kappa = v/(2*self.v0)
        beta = 2*self.alphaX*self.w*c0**2/v**2
        return self.iden*classics.sigma_combined(kappa, beta, mode=self.mode, sign=self.sign) 

# We monkey patch the mode and sign here for access later
CLASSIC.mode = 'V'
CLASSIC.sign = 'repulsive'
