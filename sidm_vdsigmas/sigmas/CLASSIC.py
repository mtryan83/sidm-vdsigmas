
from unyt import speed_of_light as c0

import cross_sections as classics

from sidm_vdsigmas.interaction import Interaction

class CLASSIC(Interaction):
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
        sig0 = self.sigma_nd(self.v0 / 50)
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
        sign = '+' if cls.sign == 'attractive' else '-'
        return f'CLASSIC{sign}{cls.mode}'
        
    def __call__(self,v):
        return self.sigscale * self.sigma_nd(v)

    def hat(self,x):
        kappa = x/2
        beta = 2*self.alphaX/self.w / x**2
        return self.iden*classics.sigma_combined(kappa, beta, mode=self.mode, sign=self.sign)

    def sigma_nd(self,v):
        kappa = v/(2*self.v0)
        beta = 2*self.alphaX*self.w*c0**2/v**2
        return self.iden*classics.sigma_combined(kappa, beta, mode=self.mode, sign=self.sign) 
    
CLASSIC.mode = 'V'
CLASSIC.sign = 'repulsive'