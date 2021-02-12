from Wavefunctions import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2
from scipy.linalg import expm

from itertools import product

"""
Generate generalized Gell-Mann matrices.
Taken from:
https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py

Made by Jonathan Gross <jarthurgross@gmail.com>
"""

def gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d.

    According to the convention in *Bloch Vectors for Qubits* by Bertlmann and
    Krammer (2008), returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`, :math:`\Lambda^{jk}_a`
    for :math:`1\leq j<k\leq d`, and :math:`I` for :math:`j=k=d`.

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
        A genereralized Gell-Mann matrix.

    """

    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
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
    r"""Return a basis of operators.

    The basis is made up of orthogonal Hermitian operators on a Hilbert space
    of dimension d, with the identity element in the last place.

    Parameters
    ----------
    d : int
        The dimension of the Hilbert space.

    Returns
    -------
    list of numpy.array
        The basis of operators.

    """
    return [gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)]


class Nucleus(Wavefunction):
    """
    Nucleus wavefunction class (2 colors)

    Inherits the following methods with modification from Wavefunction:
    (see that class for full documentation)

        colorChargeField()
        gaugeField()

    Implements the following methods:

        constructor - wrapper of Wavefunction.__init__ which also creates the basis set of matrices

        wilsonLine() - Returns the calculated Wilson Line for the nucleus
        return: np.array([N, N, n, n])

        adjointWilsonLine() - returns the Wilson Line in the adjoint representation
        return: np.array([n**2, n**2, N, N])
        
    """

    _wilsonLine = None
    _adjointWilsonLine = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _wilsonLineExists = False
    _adjointWilsonLineExists = False

    def __init__(self, colorCharges, N, delta, mu, fftNormalization=None, M=.5, g=1):
        super().__init__(N, delta, mu, colorCharges, fftNormalization, M, g) # Super constructor
        self._basis = get_basis(colorCharges)

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
                        #V = sum([self._wilsonLine[k,i,j]*self._pauli[k] for k in range(self.gluonDOF+1)])
                        V = self._wilsonLine[i,j]
                        Vdag = np.conjugate(np.transpose(V))
                        self._adjointWilsonLine[a,b,i,j] = .5 * np.trace(np.dot(np.dot(self._basis[a], V), np.dot(self._basis[b], Vdag)))


        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine

class Proton(Wavefunction):
    """
    Proton wavefunction class

    Only real difference between the super class is that the color charge field is scaled by a centered gaussian
    with a width equal to radius
    """

    def __init__(self, numColors, N, delta, mu, radius, fftNormalization=None, M=.5, g=1):
        super().__init__(N, delta, mu, numColors, fftNormalization, M, g) # Super constructor
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
