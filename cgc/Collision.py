from .Wavefunction import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2

import numba

class Collision():

    targetWavefunction = None # Implements wilson line
    incidentWavefunction = None # Doesn't (have to) implement wilson line

    _omega = None
    _omegaFFT = None
    _particlesProduced = None
    _momentaBins = None

    _omegaExists = False
    _omegaFFTExists = False
    _particlesProducedExists = False
    _momentaBinsExists = False

    def __init__(self, wavefunction1: Wavefunction, wavefunction2: Wavefunction):
        """
        Initialize a collision with two wavefunctions, presumably a nucleus and a proton. The first is expected to implement
        the wilson line, though as long as only one of the wavefunctions has this property, it will detect the proper one.

        In the case that both wavefunctions implement the wilson line, the first (wavefunction1) will be used as such.

        In the case that neither implement the wilson line, an exception will be raised.

        Parameters
        ----------
        wavefunction1 : Wavefunction (or child)
            The first wavefunction

        wavefunction2 : Wavefunction (or child)
            The second wavefunction
        """

        # Make sure that at least one has a wilson line
        wilsonLineExists1 = callable(getattr(wavefunction1, "wilsonLine", None))
        wilsonLineExists2 = callable(getattr(wavefunction2, "wilsonLine", None))

        if not wilsonLineExists1 and not wilsonLineExists2:
            raise Exception("Neither of the wavefunctions passed to Collision(Wavefunction, Wavefunction) implement the wilsonLine() method; at least one is required to.")

        if wilsonLineExists1 and not wilsonLineExists2:
            self.targetWavefunction = wavefunction1
            self.incidentWavefunction = wavefunction2
        elif wilsonLineExists2 and not wilsonLineExists1:
            self.targetWavefunction = wavefunction2
            self.incidentWavefunction = wavefunction1
        else:
            self.targetWavefunction = wavefunction1
            self.incidentWavefunction = wavefunction2

        # Make sure that both use the same number of colors
        if self.targetWavefunction.gluonDOF != self.incidentWavefunction.gluonDOF:
            raise Exception(f"Wavefunctions implement different gluon degrees of freedom (number of color charges): {self.incidentWavefunction.gluonDOF} vs. {self.targetWavefunction.gluonDOF}")

        # Probably some other checks that need to be done to make sure the two wavefunctions are compatable, but this is fine for now

        # Carry over some variables so we don't have to call through the wavefunctions so much
        self.N = self.targetWavefunction.N
        self.length = self.targetWavefunction.length
        self.gluonDOF = self.targetWavefunction.gluonDOF
        self.delta = self.targetWavefunction.delta
        self.delta2 = self.targetWavefunction.delta2
        #print(self.targetWavefunction)
        #print(self.incidentWavefunction)

        
        # Variables to do with binning the momenta later on
        self.binSize = 4*np.pi/self.length
        self.kMax = 1/self.delta
        self.numBins = int(self.kMax/self.binSize)

    def omega(self):
        """
        Calculate the field omega at each point on the lattice.

        If the field already exists, it is simply returned and no calculation is done.

        Returns
        -------
        numpy.array : shape=(2, 2, `colorCharges`**2 - 1, N, N)

        """

        if self._omegaExists:
            return self._omega

        # 2,2 is for the 2 dimensions, x and y
        self._omega = np.zeros([2, 2, self.gluonDOF, self.N, self.N], dtype='complex') # 2 is for two dimensions, x and y

        def x_deriv(matrix, i, j):
            return (matrix[i,(j+1)%self.N] - matrix[i,j-1]) / (2*self.delta)

        def y_deriv(matrix, i, j):
            return (matrix[(i+1)%self.N,j] - matrix[i-1,j]) / (2*self.delta)

        derivs = [x_deriv, y_deriv]

        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.gluonDOF):
                    for l in range(2): # 2 is number of dimensions
                        for n in range(2): # 2 is number of dimensions
                            self._omega[l,n,k,i,j] = np.sum([derivs[l](self.incidentWavefunction.gaugeField()[m], i, j) * derivs[n](self.targetWavefunction.adjointWilsonLine()[k, m], i, j) for m in range(self.gluonDOF)])

        return self._omega


    def omegaFFT(self):
        """
        Compute the fourier transform of the field omega on the lattice.

        If the fft of the field already exists, it is simply returned and no calculation is done.
        """
        if self._omegaFFTExists:
            return self._omegaFFT

        # Make sure omega exists
        self.omega()

        self._omegaFFT = fft2(self._omega, axes=(-2, -1), norm='backward')
        self._omegaFFTExists = True

        return self._omegaFFT


    def momentaBins(self):
        """
        Compute the range of momenta at which particles will be created based on the dimensions of the lattice.

        If the bins already exist, they are simply returned and no calculation is done.
        """

        if self._momentaBinsExists:
            return self._momentaBins

        self._momentaBins = [i*self.binSize for i in range(self.numBins)]
        self._momentaBinsExists = True

        return self._momentaBins

    def particlesProduced(self):
        """
        Compute the number of particles produced at each value of momentum in `momentaBins()`.
        
        If the calculation has already been done, the result is simply returned and is not repeated.
        """
        if self._particlesProducedExists:
            return self._particlesProduced

        momentaComponents, theta = _calculateMomentaOpt(self.N, self.delta)

        momentaMagSquared = np.linalg.norm(momentaComponents, axis=2)**2

        particleProductionZeroHarmonic = _calculatedNd2kHarmonicOpt(self.N, self.gluonDOF, momentaMagSquared, self.omegaFFT())

        # Now take the ratio of the two harmonics
        #particleProduction = np.abs(particleProductionSecondHarmonic) / np.abs(particleProductionZeroHarmonic)
        particleProduction = particleProductionZeroHarmonic

        vectorizedParticles = np.reshape(particleProduction, [self.N*self.N])
        vectorizedTheta = np.reshape(theta, [self.N*self.N])
        vectorizedMomentaMag = np.reshape(np.sqrt(momentaMagSquared), [self.N*self.N])
        
        self._particlesProduced = np.zeros(self.numBins)
        self.momentaBins()

        desiredHarmonic = 2

        for i in range(self.numBins):
            # Find which places on the lattice fall into this particular momentum bin
            # Note the use of element-wise (or bitwise) and, "&"
            particlesInRing = vectorizedParticles[(vectorizedMomentaMag < self.binSize*(i+1)) & (vectorizedMomentaMag > self.binSize*i)]
            thetaInRing = vectorizedTheta[(vectorizedMomentaMag < self.binSize*(i+1)) & (vectorizedMomentaMag > self.binSize*i)]

            # Zeroth harmonic is just the mean the particle values on the ring, since the argument of the exponential
            # will always be zero
            zerothHarmonicParticlesInRing = np.sum(particlesInRing) / self.N

            # Any other harmonic is a little more complicated, since the divisor is now a sum of complex
            # exponentials
            secondHarmonicParticlesInRing = particlesInRing * np.exp(1.j * desiredHarmonic * thetaInRing)
            secondHarmonicParticlesInRing = np.sum(secondHarmonicParticlesInRing) / np.sum(np.exp(1.j * desiredHarmonic * thetaInRing))

            # Now take the ratio of the second harmonic to the zeroth
            self._particlesProduced[i] = np.abs(secondHarmonicParticlesInRing) / np.abs(zerothHarmonicParticlesInRing)

        self._particlesProducedExists = True

        return self._particlesProduced

@numba.jit(nopython=True, cache=True)
def _calculateMomentaOpt(N, delta):
    """
    Optimized (via numba) function to calculated the position (momentum) in Fourier space of each point

    Parameters
    ----------

    N : int
        Size of the lattice

    delta : double
        Spacing between each point

    Returns
    -------
    (momentaComponents, theta)

    momentaComponents : array(N, N, 2)
        x and y components of the momentum at each point

    theta : array(N, N)
        Relationship between x and y components at each point, or atan2(k_y, k_x)

    """
    momentaComponents = np.zeros((N, N, 2))
    theta = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # Note that these components are of the form:
            # k_x = 2/a sin(k_x' a / 2)
            # Though the argument of the sin is simplified a bit
            momentaComponents[i,j] = [2/delta * np.sin(np.pi*i/N), 2/delta * np.sin(np.pi*j/N)]
            theta[i,j] = np.arctan2(momentaComponents[i,j,1], momentaComponents[i,j,0])

    return momentaComponents, theta

@numba.jit(nopython=True, cache=True)
def _calculatedNd2kHarmonicOpt(N, gluonDOF, momentaMagSquared, omegaFFT):
    """
    Optimized (via numba) function to calculate a given harmonic of dN/d^2k

    Parameters
    ----------

    N : int
        The system size

    gluonDOF : int
        The number of gluon degrees of freedom ((possible color charges)^2 - 1)

    momentaMagSquared : array(N, N)
        The magnitude of the momentum at each point, likely calculated (in part) with _calculateMomentaOpt()

    omegaFFT : array(2, 2, gluonDOF, N, N)
        Previously calculated omega array

    Returns
    -------
    particleProduction : array(N, N)
        The number of particles produced at each point on the momentum lattice

    """
    # Where we will calculate dN/d^2k 
    particleProduction = np.zeros((N,N))

    # # 2D Levi-Cevita symbol
    LCS = np.array([[0,1],[-1,0]])

    # # 2D Delta function
    KDF = np.array([[1,0],[0,1]])

    for y in range(N):
        for x in range(N):
            # To prevent any divide by zero errors
            if momentaMagSquared[y,x] == 0:
                continue
            
            # All of these 2s are for our two dimensions, x and y
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        for m in range(2):

                            for a in range(gluonDOF):
                                particleProduction[y,x] += np.real(2/(2*np.pi)**3 / momentaMagSquared[y,x] * (
                                    (KDF[i,j]*KDF[l,m] + LCS[i,j]*LCS[l,m])) * (
                                        omegaFFT[i,j,a,y,x] * np.conj(omegaFFT[l,m,a,y,x])))

    return particleProduction
