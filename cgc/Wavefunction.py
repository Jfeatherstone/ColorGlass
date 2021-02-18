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

    def __init__(self, colorCharges, N, delta, mu, fftNormalization=None, M=.5, g=1):
        r"""

        Base wavefunction class inplementing a generic color charge field and gauge field
        for a system with an arbitrary number of colors.

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

        fftNormalization : None | "backward" | "ortho" | "forward"
            Normalization procedure used when computing fourier transforms; see [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html) for more information

        M : float
            Experimental parameter in the laplace equation for the gauge field

        g : float
            Parameter in the laplace equation for the gauge field
        """

        self.N = N
        self.delta = delta
        self.mu = mu
        self.colorCharges = colorCharges
        self.gluonDOF = colorCharges**2 - 1
        self.fftNormalization = fftNormalization
        self.M = M
        self.g = g

        # Calculate some simple system constants
        self.length = self.N * self.delta
        self.gaussianWidth = self.mu / self.delta
        self.xCenter = self.yCenter = self.length / 2


    def colorChargeField(self):
        """
        Generates the color charge density field according to a gaussian distribution.

        If the field already exists, it is simply returned and no calculation is done.
        """
        if self._colorChargeFieldExists:
            return self._colorChargeField

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = np.random.normal(scale=self.gaussianWidth, size=(self.gluonDOF, self.N, self.N))
        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        return self._colorChargeField


    def gaugeField(self):
        """
        Calculates the gauge field for the given color charge distribution by solving the (modified)
        Poisson equation involving the color charge field using Fourier method.

        If the field already exists, it is simply returned and no calculation is done.
        """
        if self._gaugeFieldExists:
            return self._gaugeField

        # Make sure the charge field has already been generated (if not, this will generate it)
        self.colorChargeField()

        # Compute the fourier transform of the charge field
        chargeDensityFFTArr = fft2(self._colorChargeField, axes=(-2,-1), norm=self.fftNormalization)

        # This function calculates the individual elements of the gauge field in fourier space,
        # which we can then ifft back to get the actual gauge field
        # This expression was acquired by 
        def AHat_mn(m, n, chargeFieldFFT_mn):
            numerator = -self.delta**2 * self.g * chargeFieldFFT_mn
            denominator = 2 * (np.cos(2*np.pi*m*self.delta/self.length) + np.cos(2*np.pi*n*self.delta/self.length) - 2 - (self.M * self.delta)**2 / 2)
            if denominator == 0:
                return 0
            return numerator / denominator
        vec_AHat_mn = np.vectorize(AHat_mn)

        # For indexing along the lattice
        iArr = np.arange(0, self.N)
        jArr = np.arange(0, self.N)

        # Calculate the individual elements of the gauge field in fourier space
        gaugeFieldFFTArr = np.zeros_like(self._colorChargeField, dtype='complex')

        for k in range(self.gluonDOF):
            gaugeFieldFFTArr[k] = [vec_AHat_mn(i, jArr, chargeDensityFFTArr[k,i,jArr]) for i in iArr]

        # Take the inverse fourier transform to get the actual gauge field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(-2, -1), norm=self.fftNormalization))
        # Make sure this process isn't repeated unnecessarily by denoting that it has been done
        self._gaugeFieldExists = True

        return self._gaugeField


class Proton(Wavefunction):
    """
    Only real difference between the super class is that the color charge field is scaled by a centered gaussian
    with a width equal to `radius`.
    """

    def __init__(self, colorCharges, N, delta, mu, radius, fftNormalization=None, M=.5, g=1):
        """
        Wrapper for Wavefunction.__init__ that stores the radius parameter for use in generating the
        color charge field.

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

        fftNormalization : None | "backward" | "ortho" | "forward"
            Normalization procedure used when computing fourier transforms; see [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html) for more information

        M : float
            Experimental parameter in the laplace equation for the gauge field

        g : float
            Parameter in the laplace equation for the gauge field

        """

        super().__init__(colorCharges, N, delta, mu, fftNormalization, M, g) # Super constructor
        self.radius = radius

    def colorChargeField(self):
        """
        Generates the color charge density field according to a gaussian distribution, which decays according
        to another gaussian distribution.

        If the field already exists, it is simply returned and no calculation is done.
        """
        if self._colorChargeFieldExists:
            return self._colorChargeField

        # Centered gaussian distribution defined by class parameters
        def gaussian(x, y, r=self.radius, xc=self.xCenter, yc=self.yCenter):
            return np.exp( - ((x - xc)**2 + (y - yc)**2) / (2*r**2))

        protonGaussianCorrection = np.array([gaussian(i*self.delta, np.arange(0, self.N)*self.delta) for i in np.arange(0, self.N)])

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = np.random.normal(scale=self.gaussianWidth, size=(self.gluonDOF, self.N, self.N))
        self._colorChargeField *= protonGaussianCorrection # Apply the correction

        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        return self._colorChargeField
