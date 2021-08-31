import numpy as np
from scipy.fft import ifft2, fft2

"""
Some basic conventions used in this file:

Class variables that begin with an underscore ("_") are not meant to be accessed outside of the class.
They either have proper access methods (eg. gaugeField() for _gaugeField) or shouldn't be used outside
in the first place.
"""

class Wavefunction():

    _colorChargeField = None
    _gaugeField = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _colorChargeFieldExists = False
    _gaugeFieldExists = False
    #_wilsonLineExists = False # May not be implemented depending on nucleus/proton
    #_adjointWilsonLineExists = False # May not be implemented depending on nucleus/proton

    def __init__(self, colorCharges, N, delta, mu, M=.5, g=1, rngSeed=None):
        r"""

        Base wavefunction class inplementing a generic color charge and gauge field
        in an arbitrary special unitary group, \(SU\)(`colorCharges`).

        Parameters
        ----------
        colorCharges : positive integer
            The number of possible color charges; also the dimensionality of the special unitary group

        N : positive integer
            The size of the square lattice to simulate

        delta : positive float
            The distance between adjacent lattice sites

        mu : positive float
            The scaling for the random gaussian distribution that generates the color charge density

        M : float (default=.5)
            Infrared regulator parameter to regularize the Poisson equation for the gauge field.

        g : float (default=1)
            Parameter in the Poisson equation for the gauge field

        rngSeed : int (default=None)
            Seed for the random number generator to initialize the color charge field
        """

        self.N = N
        self.delta = delta
        self.mu = mu
        self.colorCharges = colorCharges
        self.gluonDOF = colorCharges**2 - 1
        self.M = M
        self.g = g

        # Precompute squares of values, since they are used often
        self.N2 = N**2
        self.M2 = M**2
        self.delta2 = delta**2

        # Calculate some simple system constants
        self.length = self.N * self.delta
        self.gaussianWidth = self.mu / self.delta
        self.xCenter = self.yCenter = self.length / 2
        self.poissonReg = (self.M2 * self.delta2) / 2

        # Setup the random number generator
        if rngSeed != None:
            self.rng = np.random.default_rng(rngSeed)
        else:
            self.rng = np.random.default_rng()

    def colorChargeField(self, forceCalculate=False):
        r"""
        Generates the color charge density 2-d field according to a gaussian distribution.
        The width of the gaussian is given by `mu` divided by the discretization `delta`. 

        If the field already exists, it is simply returned and no calculation is done.


        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        Returns
        -------

        colorChargeField : array(`colorCharges`**2 - 1, N, N)
        """
        if self._colorChargeFieldExists and not forceCalculate:
            return self._colorChargeField

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = self.rng.normal(scale=self.gaussianWidth, size=(self.gluonDOF, self.N, self.N))
        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        return self._colorChargeField


    def gaugeField(self, forceCalculate=False):
        r"""
        Calculates the gauge field for the given color charge distribution by solving the (modified)
        Poisson equation involving the color charge field

        $$g \frac{1  } {\partial_\perp^2 - m^2 } \rho_a(i^-, \vec {i}_\perp )$$

        via Fourier method.

        If the field already exists, it is simply returned and no calculation is done.


        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        Returns
        -------

        gaugeField : array(`colorCharges`**2 - 1, N, N)
        """
        if self._gaugeFieldExists and not forceCalculate:
            return self._gaugeField

        # Make sure the charge field has already been generated (if not, this will generate it)
        self.colorChargeField()

        # Compute the fourier transform of the charge field
        chargeDensityFFTArr = fft2(self._colorChargeField, axes=(-2,-1), norm='backward')

        # Absorb the numerator constants in the equation above into the charge density
        chargeDensityFFTArr = -self.delta2 * self.g / 2 * chargeDensityFFTArr

        # This function calculates the individual elements of the gauge field in fourier space,
        # which we can then ifft back to get the actual gauge field
        def AHat_mn(m, n, chargeDensityFFT_mn):
            return chargeDensityFFT_mn/(2 - np.cos(2*np.pi/self.N * m) - np.cos(2*np.pi/self.N * n) + self.poissonReg)

        # Vectorize to make it a bit faster
        # Note that in the nucleus case, we use a separate library (numba) to speed up calculations since we often
        # have to do it ~100 times, but here simple stuff should be fine
        vec_AHat_mn = np.vectorize(AHat_mn)

        # For indexing along the lattice
        iArr = np.arange(0, self.N)
        jArr = np.arange(0, self.N)

        # Calculate the individual elements of the gauge field in fourier space
        gaugeFieldFFTArr = np.zeros_like(self._colorChargeField, dtype='complex')

        for k in range(self.gluonDOF):
            gaugeFieldFFTArr[k] = [vec_AHat_mn(i, jArr, chargeDensityFFTArr[k,i,jArr]) for i in iArr]

        # Take the inverse fourier transform to get the actual gauge field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(-2, -1), norm='backward'))
        # Make sure this process isn't repeated unnecessarily by denoting that it has been done
        self._gaugeFieldExists = True

        return self._gaugeField


class Proton(Wavefunction):
    """
    Dilute object to be used in an instance of `cgc.Collision.Collision`.

    Only real difference between the super class is that the color charge field is scaled by a centered gaussian
    with a width equal to `radius`.
    """

    def __init__(self, colorCharges, N, delta, mu, radius, M=.5, g=1, rngSeed=None):
        """

        Parameters
        ----------
        colorCharges : positive integer
            The number of possible color charges; also the dimensionality of the special unitary group

        N : positive integer
            The size of the square lattice to simulate

        delta : positive float
            The distance between adjacent lattice sites

        mu : positive float
            The scaling for the random gaussian distribution that generates the color charge density

        radius : positive float
            The scaling parameter for the gaussian factor that modifies the color charge density

        M : float (default=.5)
            Infrared regulator parameter to regularize the Poisson equation for the gauge field.

        g : float (default=1)
            Parameter in the Poisson equation for the gauge field

        rngSeed : int (default=None)
            Seed for the random number generator to initialize the color charge field

        """

        super().__init__(colorCharges, N, delta, mu, M, g, rngSeed) # Super constructor
        self.radius = radius

    def colorChargeField(self, forceCalculate=False):
        r"""
        Generates the color charge density field according to a gaussian distribution, which decays according
        to a centered (different) gaussian distribution. That is, the field \(\rho\) satisfies:

        $$ \langle \rho_{a}^{(p)}(\vec x_{\perp} ) \rho_{b}^{(p)}(\vec y_{\perp} ) \rangle = g^2\mu_p^2 ~\exp\left( -\frac{\vec x_{\perp}^{2}}{2R_{p}^2} \right) ~\delta_{ab}~\delta^{(2)}(\vec x_{\perp}-\vec y_{\perp}) $$

        If the field already exists, it is simply returned and no calculation is done.


        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        Returns
        -------

        colorChargeField : array(`colorCharges`**2 - 1, N, N)
        """
        if self._colorChargeFieldExists and not forceCalculate:
            return self._colorChargeField

        # Centered gaussian distribution defined by class parameters
        def gaussian(x, y, r=self.radius, xc=self.xCenter, yc=self.yCenter):
            return np.exp( - ((x - xc)**2 + (y - yc)**2) / (2*r**2))

        protonGaussianCorrection = np.array([gaussian(i*self.delta, np.arange(0, self.N)*self.delta) for i in np.arange(0, self.N)])

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = self.rng.normal(scale=self.gaussianWidth, size=(self.gluonDOF, self.N, self.N))
        self._colorChargeField *= protonGaussianCorrection # Apply the correction

        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        return self._colorChargeField
