from .Wavefunction import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2
from scipy.linalg import expm

from itertools import product

"""
These first two methods, outside of both classes, are used to generate a set of
basis matrices used in calculating the Wilson Line

Modified version of:
https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py
originally made by Jonathan Gross
"""

def gen_gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d according to the
    convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008)

    Adapted from Jonathan Gross's [pysme library](https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py).

    Parameters
    ----------
    j : positive integer
        Index for generalized Gell-Mann matrix
    k : positive integer
        Index for generalized Gell-Mann matrix
    d : positive integer
        Dimension of the generalized Gell-Mann matrix

    Returns
    -------
    numpy.array
        A genereralized Gell-Mann matrix

    """

    if j > k:
        gjkd = np.zeros((d, d), dtype='complex')
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype='complex')
        gjkd[j - 1][k - 1] = -1.j
        gjkd[k - 1][j - 1] = 1.j
    elif j == k and j < d:
        gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
    else:
        gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

    return gjkd

def get_basis(d):
    r"""Return a Hermitian and traceless set of basis matrices for SU(d), as well
    as the identity.

    The basis is made up of d^2 - 1 generalized Gell-Mann matrices, and then the identity
    as the last matrix.

    Parameters
    ----------
    d : positive integer
        The dimension of the Hilbert space

    Returns
    -------
    list of numpy.array
        The basis

    """

    return [gen_gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)]


class Nucleus(Wavefunction):

    # Upon calling wilsonLine() or adjointWilsonLine(), these are properly defined
    _wilsonLine = None
    _adjointWilsonLine = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _wilsonLineExists = False
    _adjointWilsonLineExists = False

    def __init__(self, colorCharges, N, delta, mu, fftNormalization=None, M=.5, g=1):
        r"""
        Extension of Wavefunction that implements the calculation of the Wilson Line.

        Constructor
        -----------

        Wrapper for `super.__init__` that also creates the appropriate basis for the special
        unitary group SU(`colorCharges`)

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

        super().__init__(colorCharges, N, delta, mu, fftNormalization, M, g) # Super constructor
        self._basis = get_basis(colorCharges)

    def wilsonLine(self):
        """
        Calculate the Wilson line using the gauge field and the appropriate basis matrices.

        If the line already exists, it is simply returned and no calculation is done.
        """
        if self._wilsonLineExists:
            return self._wilsonLine

        # Make sure the gauge field has already been calculated
        if not self._gaugeFieldExists:
            self.gaugeField()

        #self._wilsonLine = np.zeros([self.gluonDOF+1, self.N, self.N], dtype='complex')
        self._wilsonLine = np.zeros([self.N, self.N, self.colorCharges, self.colorCharges], dtype='complex')

        for i in range(self.N):
            for j in range(self.N):
                # Numerical form for SU(n)
                # Note that identity is last in the _basis matrix set, so we no longer need to +1
                self._wilsonLine[i,j] = expm(1.j*sum([self._gaugeField[k,i,j]*self._basis[k] for k in range(self.gluonDOF)]))

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
                        self._adjointWilsonLine[a,b,i,j] = .5 * np.trace(np.dot(np.dot(self._basis[a], V), np.dot(self._basis[b], Vdag)))


        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine
