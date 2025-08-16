import numpy as np

class Moller(Interaction):
    name = 'MøllerV'
    file_name = name
    def __call__(self,v):
        w = self.v0
        sigma0 = self.sigconst
        return sigma0 * self.hat(v/w)

    def hat(self,x):
        x = np.maximum(x,1/50)
        return (3/(x**8 * (1 + x**-2)) * 
                                   (2*(5 + x**4 + 5*x**2) * np.log(1 + x**2) - 5*(x**4 + 2*x**2)))