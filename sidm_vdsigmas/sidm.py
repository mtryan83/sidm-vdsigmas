import numpy as np

class SIDM:
    
    def __init__(self,**kwargs):
        self.mX = None
        self.mphi = None
        self.alphaX = None
        self.w = None
        for k,v in kwargs.items():
            self.__dict__[k] = v
        self.check_consistency()

    def check_consistency(self):
        # should probably use hasattr and getattr
        sd = self.__dict__
        if 'w' in sd and ('mphi' not in sd or sd['mphi'] is None):
            self.mphi = self.w*self.mX
        elif 'mphi' in sd and ('w' not in sd or sd['w'] is None):
            self.w = self.mphi/self.mX
        for a in ['mX','mphi','alphaX','w']:
            if not hasattr(self,a) or getattr(self,a) is None:
                raise ValueError(f'Attribute {a} not provided!')
                
    def __repr__(self):
        phiunit = 'GeV/c**2' if np.log10(self.mphi.to('MeV/c**2'))>1 else 'MeV/c**2'
        return f'SIDM(mχ={self.mX:.4},mϕ={self.mphi.to(phiunit):.4},w={self.w.v:.4},α={self.alphaX})'

