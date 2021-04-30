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

    def __init__(self, colorCharges, N, delta, mu, fftNormalization=None, M=.5, g=1, longitudinalLayers=100):
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
        self.Ny = longitudinalLayers
        # Modify the gaussian width to account for the multiple longitudinal layers
        self.gaussianWidth = self.mu / self.delta / np.sqrt(self.Ny)


    def colorChargeField(self):
        """
        Generates the color charge density field according to a gaussian distribution. Differs
        from super class implementation in that it generates the numerous fields according
        to `longitudinalLayers`.

        If the field already exists, it is simply returned and no calculation is done.
        """
        if self._colorChargeFieldExists:
            return self._colorChargeField

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = np.random.normal(scale=self.gaussianWidth, size=(self.Ny, self.gluonDOF, self.N, self.N))
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

        # Note here that we have to solve the gauge field for each layer and for each gluon degree of freedom
        for l in range(self.Ny):
            for k in range(self.gluonDOF):
                gaugeFieldFFTArr[l,k] = [vec_AHat_mn(i, jArr, chargeDensityFFTArr[l,k,i,jArr]) for i in iArr]

        # Take the inverse fourier transform to get the actual gauge field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(-2, -1), norm=self.fftNormalization))
        # Make sure this process isn't repeated unnecessarily by denoting that it has been done
        self._gaugeFieldExists = True

        return self._gaugeField


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
        self._wilsonLine = np.ones([self.N, self.N, self.colorCharges, self.colorCharges], dtype='complex')

        # We now combine all of the longitudinal layers into the single wilson line
        # The path ordered exponential becomes just a product of exponentials for each layer
        for l in range(self.Ny):
            for i in range(self.N):
                for j in range(self.N):
                    # Numerical form for SU(n)
                    # Note that identity is last in the _basis matrix set, so we no longer need to +1
                    self._wilsonLine[i,j] *= expm(-1.j*sum([self._gaugeField[l,k,i,j]*self._basis[k] for k in range(self.gluonDOF)]))

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
