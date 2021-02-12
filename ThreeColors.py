from Wavefunctions import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2
from scipy.linalg import expm

class Nucleus(Wavefunction):
    """
    Nucleus wavefunction class

    Inherits the following methods with modification from Wavefunction:
    (see that class for full documentation)

        colorChargeField()
        gaugeField()

    Implements the following methods:

        constructor - wrapper of Wavefunction.__init__ with colorCharges=3

        wilsonLine() - Returns the calculated Wilson Line for the nucleus
        return: np.array([N, N, 3, 3])

        adjointWilsonLine() - returns the Wilson Line in the adjoint representation
        return: np.array([9, 9, N, N])
        
    """

    _wilsonLine = None
    _adjointWilsonLine = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _wilsonLineExists = False
    _adjointWilsonLineExists = False

    # The Gell-Mann matrices, for use in calculating the adjoint representation of the wilson line
    # specific to using 3 color charges
    # First entry is the identity, latter 8 are the proper Gell-Mann matrices
    _gell_mann = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                           [[0, -1.j, 0], [1.j, 0, 0], [0, 0, 0]],
                           [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                           [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
                           [[0, 0, -1.j], [0, 0, 0], [1.j, 0, 0]],
                           [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 0, -1.j], [0, 1, 0]],
                           [[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(3), 0], [0, 0, -2/np.sqrt(3)]]
                          ], dtype='complex')


    def __init__(self, N, delta, mu, fftNormalization=None, M=.5, g=1):
        """
        Wrapper constructor, which calls Wavefunction.__init__ with colorCharges=3
        """
        super().__init__(N, delta, mu, 3, fftNormalization, M, g) # Super constructor with colorCharges=3


    def wilsonLine(self):
        """
        Calculate the Wilson line using the gauge field and the Pauli/Gell-Mann matrices

        If the calculation has already been done, it is simply returned and no new calculation is done
        """

        if self._wilsonLineExists:
            return self._wilsonLine

        # Make sure the gauge field has already been calculated
        if not self._gaugeFieldExists:
            self.gaugeField()

        # We have a 3x3 matrix at each lattice point
        self._wilsonLine = np.zeros([self.N, self.N, 3, 3], dtype='complex')

        for i in range(self.N):
            for j in range(self.N):
                # Numerical form for SU(n)
                self._wilsonLine[i,j] = expm(1.j*sum([self._gaugeField[k,i,j]*self._gell_mann[k+1] for k in range(self.gluonDOF)]))

        self._wilsonLineExists = True

        return self._wilsonLine


    def adjointWilsonLine(self):
        """
        Calculate the Wilson line in the adjoint representation

        If the calculation has already been done, it is simply returned and no new calculation is done
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
                        V = self._wilsonLine[i,j]
                        Vdag = np.conjugate(np.transpose(V))
                        self._adjointWilsonLine[a,b,i,j] = .5 * np.trace(np.dot(np.dot(self._gell_mann[a], V), np.dot(self._gell_mann[b], Vdag)))


        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine


class Proton(Wavefunction):
    """
    Proton wavefunction class

    Only real difference between the super class is that the color charge field is scaled by a centered gaussian
    with a width equal to the parameter radius
    """

    def __init__(self, N, delta, mu, radius, fftNormalization=None, M=.5, g=1):
        """
        Wrapper constructor, which calls Wavefunction.__init__ with colorCharges=3 and saves the
        new parameter, radius
        """
        super().__init__(N, delta, mu, 3, fftNormalization, M, g) # Super constructor with colorCharges=3
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
