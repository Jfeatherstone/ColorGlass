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

    # The pauli matrices, for use in calculating the adjoint representation of the wilson line
    # specific to using 2 color charges
    _pauli = np.array([np.array([[1, 0], [0, 1]], dtype='complex'),
                      np.array([[0, 1], [1, 0]], dtype='complex'),
                      np.array([[0, -1.j], [1.j, 0]], dtype='complex'),
                      np.array([[1., 0], [0, -1.]], dtype='complex')])

    def __init__(self, N, delta, mu, fftNormalization=None, M=.5, g=1):
        r"""
        Constructor
        -----------

        Wrapper for `super.__init__` with `colorCharges` = 2.

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

        super().__init__(2, N, delta, mu, fftNormalization, M, g) # Super constructor with colorCharges=2

    def wilsonLine(self):
        """
        Calculate the Wilson line using the gauge field and the Pauli matrices using
        the closed form that exists for the exponential of the Pauli matrices.

        If the line already exists, it is simply returned and no calculation is done.
        """
        if self._wilsonLineExists:
            return self._wilsonLine

        # Make sure the gauge field has already been calculated
        if not self._gaugeFieldExists:
            self.gaugeField()

        #self._wilsonLine = np.zeros([self.gluonDOF+1, self.N, self.N], dtype='complex')
        self._wilsonLine = np.zeros([self.N, self.N, 2, 2], dtype='complex')

        for i in range(self.N):
            for j in range(self.N):
                # Closed form available for SU(2)
                normA = np.sqrt(np.sum([np.sqrt(np.real(self._gaugeField[k,i,j])**2 + np.imag(self._gaugeField[k,i,j])**2) for k in range(self.gluonDOF)]))
                self._wilsonLine[i,j] = self._pauli[0]*np.cos(normA)
                for k in range(self.gluonDOF):
                    self._wilsonLine[i,j] += self._pauli[k+1]*self._gaugeField[k,i,j] * 1.j * np.sin(normA) / normA
                
                # Numerical form for SU(n)
                #self._wilsonLine[i,j] = expm(1.j*sum([self._gaugeField[k,i,j]*self._pauli[k+1] for k in range(self.gluonDOF)]))

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
                        #V = sum([self._wilsonLine[k,i,j]*self._pauli[k] for k in range(self.gluonDOF+1)])
                        V = self._wilsonLine[i,j]
                        Vdag = np.conjugate(np.transpose(V))
                        self._adjointWilsonLine[a,b,i,j] = .5 * np.trace(np.dot(np.dot(self._pauli[a], V), np.dot(self._pauli[b], Vdag)))


        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine
