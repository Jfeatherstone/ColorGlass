from .Wavefunction import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2
from scipy.linalg import expm

class Nucleus(Wavefunction):
    
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
        r"""
        Constructor
        -----------

        Wrapper for `super.__init__` with `colorCharges` = 3.

        Parameters
        ----------
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

        super().__init__(3, N, delta, mu, fftNormalization, M, g) # Super constructor with colorCharges=3


    def wilsonLine(self):
        """
        Calculate the Wilson line by numerically computing the 
        exponential of the gauge field times the Gell-mann matrices.
        Numerical calculation is done using [scipy's expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)

        If the line already exists, it is simply returned and no calculation is done.
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
        Calculate the Wilson line in the adjoint representation.

        If the line already exists, it is simply returned and no calculation is done.
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
