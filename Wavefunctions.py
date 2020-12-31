import numpy as np
from scipy.fft import ifft2, fft2

"""
Some basic conventions used in this file:

Class variables that begin with an underscore ("_") are not meant to be accessed outside of the class.
They either have proper access methods (eg. gaugeField() for _gaugeField) or shouldn't be used outside
in the first place.
"""

class Wavefunction():
    """
    Base wavefunction class inplementing a generic color charge field and gauge field
    for a system with an arbitrary number of colors
    """

    _colorChargeField = None
    _gaugeField = None
    #_wilsonLine = None # May not be implemented depending on nucleus/proton
    #_adjointWilsonLine = None # May not be implemented depending on nucleus/proton

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _colorChargeFieldExists = False
    _guageFieldExists = False
    #_wilsonLineExists = False # May not be implemented depending on nucleus/proton
    #_adjointWilsonLineExists = False # May not be implemented depending on nucleus/proton

    def __init__(self, N, delta, mu, colorCharges, fftNormalization=None, M=.5, g=1):
        """
        Initialization

        ## Keyword Arguments:

        N: The number of lattice sites in each direction

        delta: The spacing between each lattice site

        mu: Parameter relating to the gaussian distribution used to generate the charge field

        M: The experimental correction constant used in solving for A (default=.5)

        g: Another constant? (default=1)

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
        Generates the color charge density field according to a gaussian distribution 

        If the field already exists, it is simply returned and no calculation is done
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
        Calculates the gauge field for the given color charge distribution

        If the field already exists, it is simply returned and no calculation is done
        """
        if self._guageFieldExists:
            return self._gaugeField

        # Make sure the charge field has already been generated
        if not self._colorChargeFieldExists:
            self.colorChargeField()

        # Compute the fourier transform of the charge field
        chargeDensityFFTArr = fft2(self._colorChargeField, axes=(-2,-1), norm=self.fftNormalization)

        # This function calculates the individual elements of the gauge field in fourier space,
        # which we can then ifft back to get the actual gauge field
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

        # Take the inverse fourier transform to get the actual guage field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(-2, -1), norm=self.fftNormalization))
        # Make sure this process isn't repeated unnecessarily by denoting that it has been done
        self._gaugeFieldExists = True

        return self._gaugeField
