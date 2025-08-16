import abc
import numpy as np

from functools import cache

import scipy.special as special
import scipy.interpolate as interpolate

from unyt import (unyt_array, unyt_quantity, speed_of_light as c0, 
                    reduced_planck_constant as hbar, gravitational_constant as G0)
from unyt.dimensions import length,dimensionless

from .sidm import SIDM

"""
Intention is to have the majority of all cross section functions implemented in
this class. Please see :doc:`Cross_Sections` and :doc:`Tutorials` for more info
"""

sigunit = unyt_quantity(1,'cm**2/g')

class Interaction(object):
    r"""Abstract base class for creating cross sections.

    Specific cross sections can be implemented by inheriting this class. These 
    child classes must set their name and file_name  before calling this 
    constructor, and they must implement the :func:`name`, :func:`file_name`,
    :func:`__call__` and :func:`hat` methods. 

    This class provides a general implementation of the $K_n$-type functions,
    $n=-\frac{dK_{x}}{dv}$, $\frac{dn}{dv}$, the value of $x$ computed using
    the implicit equation $n(x) = n_0$, and the dimensionless and dimensionful
    versions of $\hat{\sigma}$.
    """

    def __init__(self,*,
                 m=None,mphi=None,alphaX=None,
                 sigconst=None,w=None,
                 sidm=None,
                 disable_warning=False,
                ):
        r"""Create an interaction cross section
        
        Creates an interaction cross section object from the specified
        SIDM model parameters. Model parameters can be specified in one 
        of 3 ways (ordered by priority): as an instance of the SIDM class;
        via the m, mphi, and w parameters; or via the sigconst and w 
        parameters. Any of the 3 ways will populate the other the parameters
        of the other two. 

        Inputs:
            m, mphi: unyt_quantity, optional
            Mass of the SIDM (m) or SIDM mediator (mphi) as a 
            dimensionful quantity
            alphaX: float
            SIDM fine structure constant. Defaults to 1

            sigconst: float | unyt_quantity, optional
            Constant velocity portion of the cross section, commonly denoted
            as $\sigma_0$ or $\sigma_0/m$. Units can be *either* $length^2$
            or as $length^2/mass$. If provided as a float, will assume $cm^2/g$
            w: float | unyt_quantity
            Scale velocity/mediator-mass ratio of the cross section. If units
            are provided and not dimensionless, assumed to be scale velocity, 
            aka **`v0`**. If no units or dimensionless, assumed to be 
            mphi/m ratio and `v0` is defined as $v_0 = w c$ in km/s. If neither 
            sigconst nor w have have units, **both are assumed to have units**

            sidm: SIDM, optional
            SIDM parameter class instance.  Effectively the same as providing
            m, mphi, alphaX, and w 

            disable_warning: bool, optional
            Flag to turn off the warning about neither sigconst nor w having
            units. Default False
        """
        self.sidm = None
        if sidm is not None:
            m = sidm.mX
            mphi = sidm.mphi
            alphaX = sidm.alphaX
            w = sidm.w
            self.sidm = sidm
        self.m=m
        self.mphi=mphi
        alphaX = alphaX if alphaX is not None else 1
        self.alphaX=alphaX
        if sigconst is not None and not isinstance(sigconst,(unyt_array,unyt_quantity)):
            # if sigconst and w are supplied, but *neither* have units
            # assume *both* should have units
            if w is not None and not isinstance(w,(unyt_array,unyt_quantity)):
                w = unyt_quantity(w,'km/s')
                if not disable_warning:
                    print('Neither sigconst nor w had units. Assuming both should.')
            sigconst = sigconst * sigunit
        if m is not None and mphi is not None and alphaX is not None:
            w = mphi/m
            sigconst = (hbar/c0)**2 * np.pi * alphaX**2/(w**2 * m**3)
        if isinstance(w,(unyt_array,unyt_quantity)) and w.units != dimensionless:
            self.v0 = w
            w = w/c0
        else:
            self.v0 = w*c0
        if m is None:
            self.m = (((hbar/c0)**2 * np.pi*alphaX**2/(w**2*sigconst))**(1/3)).to('GeV/c**2')
        if mphi is None:
            self.mphi = w * self.m
        self.w = w
        if self.sidm is None:
            self.sidm = SIDM(mX=self.m,mphi=self.mphi,alphaX=self.alphaX,w=self.w)
        sigconst = sigconst.to('cm**2') if sigconst.units.dimensions/length**2==1 else sigconst.to('cm**2/g')
        self.sigconst = sigconst

    @property
    @abc.abstractproperty
    def name(self):
        raise NotImplementedError('Defined by implemented class!')

    @property
    @abc.abstractproperty
    def file_name(self):
        raise NotImplementedError('Defined by implemented class!')

    @classmethod
    @cache
    def _getGLstuff(cls,n=20,alpha=3,mu=True):
        """Return (cached) roots from scipy.special General Gauss-Laguerre function"""
        return special.roots_genlaguerre(n,alpha,mu)

    def Kn(self,x_s,n=5,*,N=20):
        '''
        Gauss-Laguerre for averaging over crosssection*v^5
        p1 = alpha*c/v_s; p2 = alpha*c/w; cross_func takes inputs p1 and p2;
        
        Quadrature for Integral dx x^3 exp(-x) function(x) = Sum[wgt[i]*function(loc[i]),{i,1,N}]
        We need Integral dv v^5 v^2 exp(-v^2/(4 v_s^2)) cross(v) / (same integral without cross(v))
        This can be rewritten as Integral dx x^3 exp(-x) cross(v = 2 v_s sqrt(x)) / 6, 
        where x = (v/v_s)^2 / 4 
    
        This holds true for v^p as long as we change x^3 -> x^alpha, where alpha=(p+1)/2 and 6 -> 
        sum(wgts)=mu
        '''
        alpha = (n+1)/2
        loc,wgt,mu = Interaction._getGLstuff(n=N,alpha=alpha,mu=True) 
        cross = 0
        for x,w in zip(loc,wgt): # 2*sqrt(x) = v/v1d; p1v = alpha*c/v = p1*v1d/v = p1/(2*np.sqrt(x))
            cross += w*self.hat(2*x_s*np.sqrt(x))
        return cross/mu

    def K5(self,x_s):
        # For some reason these appear to be off. Multiplying v_s by 0.8233 helps
        return self.Kn(0.8233*x_s,n=5)

    def Keff(self,x_s):
        x_s = x_s*0.8233
        K5 = self.Kn(x_s)
        K7 = self.Kn(x_s,n=7)
        K9 = self.Kn(x_s,n=9)
        return (28*K5**2 + 80*K5*K9 - 64*K7**2)/(77*K5 - 112*K7 + 80*K9)

    def _gen_n_splines(self,):
        # generate interpolating spline
        x = np.logspace(-3,3)
        logK5 = np.log10(self.K5(x))
        logKeff = np.log10(self.Keff(x))
        logx = np.log10(x)
        K5spl = interpolate.CubicSpline(logx,logK5)
        Keffspl = interpolate.CubicSpline(logx,logKeff)
        self.dlogK5dlogx = K5spl.derivative()
        self.dlogKeffdlogx = Keffspl.derivative()
        ns,nuinds = np.unique(np.maximum(-self.dlogK5dlogx(logx),1e-5),return_index=True)
        self.xofnspl = interpolate.CubicSpline(ns,logx[nuinds])
    
    def n(self,x_s=None,*,use_K5=True,v_s=None):
        if v_s is not None:
            x_s = v_s/self.v0
        if x_s is None:
            raise TypeError('Either x_s (preferred) or v_s must be provided')
        if not hasattr(self,'dlogK5dlogx'):
            self._gen_n_splines()
        # need n = -dlogK/dlogx
        logxs = np.log10(x_s)
        n = -self.dlogK5dlogx(logxs) if use_K5 else -self.dlogKeffdlogx(logxs)
        return np.maximum(n,1e-5)

    def dndv(self,x_s,*,use_K5=True):
        if not hasattr(self,'dn5dv'):
            # generate spline derivative of n
            if not hasattr(self,'dlogk5dlox'):
                # need to generate n splines
                self.n(x_s)
            self.dn5dv = self.dlogK5dlogx.derivative()
            self.dneffdv = self.dlogKeffdlogx.derivative()
        logxs = np.log10(x_s)
        dndv = self.dn5dv(logxs) if use_K5 else self.dneffdv(logxs)
        return dndv

    def x(self,n):
        '''
        Given a value of n, find the corresponding value of x=v/w that provides it
        '''
        if not hasattr(self,'xofnspl'):
            self._gen_n_splines()
        return 10**self.xofnspl(n)

    def dim_sigma_hat(self,what,*,C=0.6):
        '''
        Eq 8 from Gad-Nasr (dimensionful cross section)
        Basically, just doesn't do the M/4*pi*r**2 scaling
        '''
        K5 = self.K5(what)
        Keff = self.Keff(what)
        sigma0om = self.sigconst
        a = 4/np.sqrt(np.pi)
        b = 25/32*np.sqrt(np.pi)
        return np.sqrt(a*C/b * K5 * Keff) * sigma0om
    
    def sigma_hat_fun(self,what,*,vn=None,rhon=None,Mn=None,rn=None,C=0.6):
        '''
        Eq 8 from Gad-Nasr (dimensionless cross section)
        '''
        part1 = self.dim_sigma_hat(what,C=C)
        if Mn is not None:
            return (part1 * (Mn/(4*np.pi*rn**2))).to('dimensionless').v
        return  (part1 * np.sqrt(rhon/(4*np.pi*G0))*vn).to('dimensionless').v

    def __repr__(self):
        return f'{self.name}(w={self.v0.to("km/s"):4.4}, σ0/m={self.sigconst.to("cm**2/g"):.5})'

    @abc.abstractmethod
    def __call__(self,v):
        raise NotImplementedError('Needs to be defined in subclass')

    @abc.abstractmethod
    def hat(self,x):
        raise NotImplementedError('Needs to be defined in subclass')

