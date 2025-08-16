import numpy as np

from sidm_vdsigmas.interaction import Interaction

class Rutherford(Interaction):
    name = 'RutherfordV'
    file_name = name
    def __call__(self,v):
        w = self.v0
        sigma0 = self.sigconst
        return sigma0 * self.hat(v/w)

    def hat(self,x):
        x = np.maximum(x,1/50)
        return 6*x**-6*(-2*x**2 + (2+x**2)*np.log(1+x**2))