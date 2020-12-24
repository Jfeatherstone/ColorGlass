from Wavefunctions import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2

class Nucleus(Wavefunction):
    """
    Nucleus wavefunction class

    Implements the calculation of the wilson line with 2 colors
    """

    _wilsonLine = None
    _adjointWilsonLine = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _wilsonLineExists = False
    _adjointWilsonLineExists = False

    # The pauli matrices, for use in calculating the adjoint representation of the wilson line
    # specific to using 2 color charges
    _pauli = np.array([np.array([[1, 0], [0, 1]], dtype='complex'),
                      np.array([[0, 1], [1, 0]], dtype='complex'),
                      np.array([[0, -1.j], [1.j, 0]], dtype='complex'),
                      np.array([[1., 0], [0, -1.]], dtype='complex')])

    def __init__(self, N, delta, mu, fftNormalization=None, M=.5, g=1):
        super().__init__(N, delta, mu, 2, fftNormalization, M, g) # Super constructor with colorCharges=2

    def wilsonLine(self):
        """
        Calculate the Wilson line using the gauge field and the Pauli/Gell-Mann matrices

        If the line already exists, it is simply returned and no calculation is done
        """
        if self._wilsonLineExists:
            return self._wilsonLine

        # Make sure the gauge field has already been calculated
        if not self._gaugeFieldExists:
            self.gaugeField()

        self._wilsonLine = np.zeros([self.gluonDOF+1, self.N, self.N], dtype='complex')

        for i in range(self.N):
            for j in range(self.N):
                normA = np.sqrt(np.sum([self._gaugeField[k,i,j]**2 for k in range(self.gluonDOF)]))
                self._wilsonLine[0,i,j] = np.cos(normA)
                for k in range(self.gluonDOF):
                    self._wilsonLine[k+1,i,j] = self._gaugeField[k,i,j] * 1.j * np.sin(normA) / normA


        self._wilsonLineExists = True

        return self._wilsonLine

    def adjointWilsonLine(self):
        """
        Calculate the Wilson line in the adjoint representation

        If the line already exists, it is simply returned and no calculation is done
        """
        if self._adjointWilsonLineExists:
            return self._adjointWilsonLine
        
        # Make sure the wilson line has already been calculated
        if not self._wilsonLineExists:
            self.wilsonLine()

        self._adjointWilsonLine = np.zeros([self.gluonDOF+1, self.gluonDOF+1, self.N, self.N], dtype='complex')

        for a in range(self.gluonDOF+1):
            for b in range(self.gluonDOF+1):
                for i in range(self.N):
                    for j in range(self.N):
                        V = sum([self._wilsonLine[k,i,j]*self._pauli[k] for k in range(self.gluonDOF+1)])
                        Vdag = np.conjugate(np.transpose(V))
                        self._adjointWilsonLine[a,b,i,j] = .5 * np.trace(np.dot(np.dot(self._pauli[a], V), np.dot(self._pauli[b], Vdag)))


        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine


class Proton(Wavefunction):
    """
    Proton wavefunction class

    Only real difference between the super class is that the color charge field is scaled by a centered gaussian
    with a width equal to radius
    """

    def __init__(self, N, delta, mu, radius, fftNormalization=None, M=.5, g=1):
        super().__init__(N, delta, mu, 2, fftNormalization, M, g) # Super constructor with colorCharges=2
        self.radius = radius

    def colorChargeField(self):
        """
        Generates the color charge density field according to a gaussian distribution

        If the field already exists, it is simply returned and no calculation is done
        """
        if self._colorChargeFieldExists:
            return self._colorChargeField

        def gaussian(x, y, r=self.radius, xc=self.xCenter, yc=self.yCenter):
            return np.exp( - ((x - xc)**2 + (y - yc)**2) / (2*r**2))

        protonGaussianCorrection = np.array([gaussian(i*self.delta, np.arange(0, self.N)*self.delta) for i in np.arange(0, self.N)])

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = np.random.normal(scale=self.gaussianWidth, size=(self.gluonDOF, self.N, self.N))
        self._colorChargeField *= protonGaussianCorrection
        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        return self._colorChargeField
