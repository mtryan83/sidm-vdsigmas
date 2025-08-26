import numpy as np

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
    
    def __init__(self,**kwargs):
        self.mX = None
        self.mphi = None
        self.alphaX = None
        self.w = None
        for k,v in kwargs.items():
            self.__dict__[k] = v
        self.check_consistency()

    def check_consistency(self):
        """ Check that all parameters are present and that w and mphi are consistent
        """
        # should probably use hasattr and getattr
        sd = self.__dict__
        if 'w' in sd and ('mphi' not in sd or sd['mphi'] is None):
            self.mphi = self.w*self.mX
        elif 'mphi' in sd and ('w' not in sd or sd['w'] is None):
            self.w = self.mphi/self.mX
        elif 'mphi' in sd and 'w' in sd:
            if not np.isclose(self.w,self.mphi/self.mX):
                raise ValueError(f'w and mphi are inconsistent: {w=:.4} mphi/mX={mphi/mX:.4}')
        for a in ['mX','mphi','alphaX','w']:
            if not hasattr(self,a) or getattr(self,a) is None:
                raise ValueError(f'Attribute {a} not provided!')
                
    def __repr__(self):
        phiunit = 'GeV/c**2' if np.log10(self.mphi.to('MeV/c**2'))>1 else 'MeV/c**2'
        return f'SIDM(mχ={self.mX:.4},mϕ={self.mphi.to(phiunit):.4},w={self.w.v:.4},α={self.alphaX})'

