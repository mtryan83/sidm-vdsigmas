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
        r"""Compute the quantity K_n using generalized Gauss-Laguerre quadrature

        See the Notes section for the implementation.

        Inputs:
            x_s: float | array
            Scaled, dimensionless velocity, e.g. v/self.v_0

            n: int, optional
            index, Default is 5

            N: int, optional
            order of the Laguerre polynomial to use, controls error. Default is 20

        Returns:
            float | array
            The quantity K_n evaluated at x_s

        Notes:
        :math:`K_n` is defined 
        .. math::
            K_n(v_{1D}) = \frac{\langle \sigma(v) v_{rel}^n\rangle_{v_{1D}}}{\lim_{w\rightarrow 0}\langle \sigma(v) v_{rel}^n\rangle_{v_{1D}}}
             
        We can compute this using Generalized Gauss-Laguerre quadrature using
        the fact that we have the following relationships:
        .. math::
            \langle g(v_{rel}) v_{rel}^n\rangle = C_1 \int_{0}^{\infty} g(v_{rel}) v_{rel}^{(p+1)/2} e^{-v_{rel}^2/(2v_{1D})^2} dv_{rel}
        
        and 
        .. math::
            \int_{0}^{\infty}f(x) x^{\alpha} e^{-x} dx \approx \sum_{i=1}^{N} w_i f(x_i)
        
        where :math:`w_i` and :math:`x_i` are the weights and roots of the
        generalized Laguerre polynomials :math:`L_N^{(\alpha)}`. Thus if we
        define :math:`x_i=\left(\frac{v_{rel}}{2 v_{1D}}\right)^2`
        :math:`\mu=\sum_{i=1}^{N}w_i`, and :math:`\alpha=\frac{p+1}{2}` and assume
        that :math:`\lim_{w\rightarrow 0} \sigma(v) = \sigma_0` such that we
        can write :math:`\sigma(v) = \sigma_0 \tilde{\sigma}(v)`, then
        .. math::
            K_n(v_{1D}) \approx \frac{1}{\mu}\sum_{i=1}^{N} w_i \tilde{\sigma}(2 v_s \sqrt{x_i})
        
        Lastly, we'll define :math:`\hat{\sigma}(x_s)=\tilder{\sigma}(v_s/v_0)`
        where :math:`v_0` is our cross section velocity scale.
        """
        alpha = (n+1)/2
        loc,wgt,mu = Interaction._getGLstuff(n=N,alpha=alpha,mu=True) 
        cross = 0
        for x,w in zip(loc,wgt): # 2*sqrt(x) = v/v1d; p1v = alpha*c/v = p1*v1d/v = p1/(2*np.sqrt(x))
            cross += w*self.hat(2*x_s*np.sqrt(x))
        return cross/mu

    def K5(self,x_s):
        """Compute the quantity K_5

        Shortcut version of :code:`K_n(x_s,n=5)`

        Inputs:
            x_s: float | array
            Scaled, dimensionless velocity, e.g. v/self.v_0

        Returns:
            float | array
            The quantity K_5 evaluated at x_s
        """
        # For some reason these appear to be off. Multiplying v_s by 0.8233 helps
        return self.Kn(0.8233*x_s,n=5)

    def Keff(self,x_s):
        r"""Compute the second order K_eff term

        Inputs:
            x_s: float | array
            Scaled, dimensionless velocity, e.g. v/self.v_0

        Returns:
            float | array
            The second order K_eff term evaluated at x_s

        Notes:
        The second order K_eff term is defined as 
        .. math::
            K_{eff}^{(2)} = \frac{28 K_5^2 + 80*K_5*K_9 - 64*K7^2}{77*K5 - 112*K7 + 80*K9}
        
        """
        x_s = x_s*0.8233
        K5 = self.Kn(x_s)
        K7 = self.Kn(x_s,n=7)
        K9 = self.Kn(x_s,n=9)
        return (28*K5**2 + 80*K5*K9 - 64*K7**2)/(77*K5 - 112*K7 + 80*K9)

    def _gen_n_splines(self,):
        """Generate and cache interpolating splines"""
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
        r"""Compute the negative log derivative of K_n(v) or K_n(x)

        Compute the negative log derivative of either K_5(x_s) or K_eff(x_s)
        using a scipy.interpolate.CubicSpline fit. The spline fits are cached.

        Inputs:
            x_s: float | array, optional
            Scaled, dimensionless velocity, e.g. v/self.v_0. Must provide one
            of x_s, v_s. If x_s is not provided, x_s = v_s/self.v0

            use_K5: bool, optional
            If True, K_n is K_5. If False, K_n is K_eff. Default True

            v_s: float | array, optional
            Velocity. If both x_s and v_s are provided, prefer v_s. Must
            provide one of x_s, v_s.

        Returns:
            float | array
            The negative log derivative of K_n evaluated at x_s
        """
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
        r"""Compute n'(x)

        Compute the second derivative of either K_5(x_s) or K_eff(x_s)
        using a scipy.interpolate.CubicSpline fit. The spline fits are cached.

        Inputs:
            x_s: float | array
            Scaled, dimensionless velocity, e.g. v/self.v_0. 

            use_K5: bool, optional
            If True, K_n is K_5. If False, K_n is K_eff. Default True

        Returns:
            float | array
            The quantity n'(x_s)
        """
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
        r"""Given a value of n, find the corresponding value of x=v/w that provides it

        Invert the definition of :math:`n(x) = -\frac{dK_5(x)}{dx}` for a given
        value of n. Uses a cached scipy.interpolate.CubicSpline of n(log(x))

        Inputs:
            n: float
            The value of n to 
        """
        if not hasattr(self,'xofnspl'):
            self._gen_n_splines()
        return 10**self.xofnspl(n)

    def dim_sigma_hat(self,what,*,C=0.6):
        """Compute the dimensionful effective cross section 
        
        Compute Eq 8 from Gad-Nasr (dimensionful cross section) but without 
        the M/4*pi*r**2 scaling. This is the effective cross section in normal
        length^2/mass units.

        Inputs:
            what: float | array
            Dimensionless velocity, defined as v/self.v0

            C: float, optional
            Calibration parameter. See Gad-Nasr for details. Default is 0.6
            
        Returns:
            unyt_quantity | unyt_array
            The dimensionfull effective cross section with the same shape as
            what
        """
        K5 = self.K5(what)
        Keff = self.Keff(what)
        sigma0om = self.sigconst
        a = 4/np.sqrt(np.pi)
        b = 25/32*np.sqrt(np.pi)
        return np.sqrt(a*C/b * K5 * Keff) * sigma0om
    
    def sigma_hat_fun(self,what,*,vn=None,rhon=None,Mn=None,rn=None,C=0.6):
        """Compute the dimensionless effective cross section 
        
        Compute the dimensionless effective cross section defined in Eq 8
        from Gad-Nasr. Two possible methods of rescaling are provided: either
        using vn and rhon (scale velocity and density) or using Mn and rn
        (scale mass and radius).

        Inputs:
            what: float | array
            Dimensionless velocity, defined as v/self.v0

            vn, rhon: unyt_quantity | unyt_array, optional
            Scale velocity and density. Must provide either vn/rhon or Mn/rn.
            If an array is provided, it must be broadcastable against what
            Common choices are v_max, rho_s from an NFW profile

            Mn, rn: unyt_quantity | unyt_array, optional
            Scale mass and radii. Must provide either vn/rhon or Mn/rn.
            If both sets are provided, the vn/rhon pair is ignored.
            If an array is provided, it must be broadcastable against what

            C: float, optional
            Calibration parameter. See Gad-Nasr for details. Default is 0.6

        Returns:
            unyt_quantity | unyt_array
            The dimensionless effective cross section with the same shape as
            what
        """
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

