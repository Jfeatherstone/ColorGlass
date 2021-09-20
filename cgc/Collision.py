from .Wavefunction import Wavefunction

import numpy as np
from scipy.fft import ifft2, fft2

import numba

CACHE_OPTIMIZATIONS = True

class Collision():

    targetWavefunction = None # Implements wilson line
    incidentWavefunction = None # Doesn't (have to) implement wilson line

    _omega = None
    _omegaFFT = None
    _particlesProduced = None
    _particlesProducedDeriv = None
    _momentaMagSquared = None
    _momentaComponents = None
    _thetaInFourierSpace = None
    _momentaBins = None
    _fourierHarmonics = None # This will be initialized as an empty dict to store harmonics (see __init__)

    _omegaExists = False
    _omegaFFTExists = False
    _momentaComponentsExist = False
    _particlesProducedExists = False
    _particlesProducedDerivExists = False
    _momentaBinsExists = False

    def __init__(self, wavefunction1: Wavefunction, wavefunction2: Wavefunction):
        r"""
        Initialize a collision with two wavefunctions, presumably a nucleus and a proton. One must implement
        the wilson line, though the order of the arguments does not matter.

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
        self.kMax = 2/self.delta
        self.numBins = int(self.kMax/self.binSize)

        # This has to be initialized as an empty dict within the constructor
        # because otherwise it can retain information across separate objects
        # (no idea how, but this fixes it)
        self._fourierHarmonics = {}

    def omega(self, forceCalculate=False, verbose=0):
        r"""
        Calculate the field omega at each point on the lattice.

        If the field already exists, it is simply returned and no calculation is done.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        omega : array(2, 2, `colorCharges`**2 - 1, N, N)

        """

        if self._omegaExists and not forceCalculate:
            return self._omega

        self.incidentWavefunction.gaugeField(verbose=verbose)
        self.targetWavefunction.adjointWilsonLine(verbose=verbose)

        if verbose > 0:
            print(f'Calculating {type(self).__name__} omega' + '.'*10, end='')

        self._omega = _calculateOmegaOpt(self.N, self.gluonDOF, self.delta, self.incidentWavefunction.gaugeField(), self.targetWavefunction.adjointWilsonLine())

        self._omegaExists = True

        if verbose > 0:
            print('finished!')

        return self._omega


    def omegaFFT(self, forceCalculate=False, verbose=0):
        r"""
        Compute the fourier transform of the field omega on the lattice.

        If the fft of the field already exists, it is simply returned and no calculation is done.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        omegaFFT : array(2, 2, `colorCharges`**2 - 1, N, N)

        """
        if self._omegaFFTExists and not forceCalculate:
            return self._omegaFFT

        # Make sure omega exists
        self.omega(verbose=verbose)

        if verbose > 0:
            print(f'Calculating {type(self).__name__} omega fourier transform' + '.'*10, end='')

        # We want to do the normalization explicitly, but scipy doesn't offer no
        # normalization as an option, so we just set it to be the opposite of whatever
        # we are doing (forward for ifft, backward for fft)
        # (we had some issues with scipy changing its default mode)
        self._omegaFFT = self.delta2 * fft2(self._omega, axes=(-2, -1), norm='backward')
        self._omegaFFTExists = True

        if verbose > 0:
            print('finished!')

        return self._omegaFFT


    def momentaBins(self, forceCalculate=False, verbose=0):
        r"""
        Compute the range of momenta at which particles will be created based on the dimensions of the lattice.

        The exact values are:

        - \( k_{max} = 2 / \Delta\)
        - \( w_k = 4 \pi / L \)

        If the bins already exist, they are simply returned and no calculation is done.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        momentaBins : array(numBins = L / (delta 2 pi))
        """

        if self._momentaBinsExists and not forceCalculate:
            return self._momentaBins

        if verbose > 0:
            print(f'Calculating {type(self).__name__} momentum bins' + '.'*10, end='')

        self._momentaBins = [i*self.binSize for i in range(self.numBins)]
        self._momentaBinsExists = True

        if verbose > 0:
            print('finished!')

        return self._momentaBins


    def momentaComponents(self, forceCalculate=False, verbose=0):
        r"""
        Compute the components of the momentum at each point on the lattice, according to:

        $$ (k_x, k_y) = \frac{2}{\Delta} \left( \sin\left( \frac{\pi i}{N} \right), \sin\left( \frac{\pi j}{N} \right) \right) $$

        where \(i\) and \(j\) index the \(x\) and \(y\) directions in real space, respectively.

        If the calculation has already been done, the result is simply returned and is not repeated.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        momentaComponents : array(N, N, 2)
        """

        if self._momentaComponentsExist and not forceCalculate:
            return self._momentaComponents

        if verbose > 0:
            print(f'Calculating {type(self).__name__} momentum components' + '.'*10, end='')

        self._momentaComponents, self._thetaInFourierSpace = _calculateMomentaOpt(self.N, self.delta)
        self._momentaMagSquared = np.linalg.norm(self._momentaComponents, axis=2)**2

        self._momentaComponentsExist = True

        if verbose > 0:
            print('finished!')

        return self._momentaComponents


    def momentaMagnitudeSquared(self, forceCalculate=False, verbose=0):
        r"""
        Compute the magnitude of the momentum at each point on the lattice, according to:

        $$ |k| = \sqrt{k_x^2 + k_y^2} $$

        $$ (k_x, k_y) = \frac{2}{\Delta} \left( \sin\left( \frac{\pi i}{N} \right), \sin\left( \frac{\pi j}{N} \right) \right) $$

        where \(i\) and \(j\) index the \(x\) and \(y\) directions in real space, respectively.

        If the calculation has already been done, the result is simply returned and is not repeated.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        momentaComponents : array(N, N)
        """

        if self._momentaComponentsExist and not forceCalculate:
            return self._momentaMagSquared

        if verbose > 0:
            print(f'Calculating {type(self).__name__} momenta magnitude squared' + '.'*10, end='') 

        self._momentaComponents, self._thetaInFourierSpace = _calculateMomentaOpt(self.N, self.delta)
        self._momentaMagSquared = np.linalg.norm(self._momentaComponents, axis=2)**2

        self._momentaComponentsExist = True

        if verbose > 0:
            print('finished!')

        return self._momentaMagSquared


    def particlesProducedDeriv(self, forceCalculate=False, verbose=0):
        r"""
        Compute the derivative of particles produced (\( \frac{d^2 N}{d^2 k} \)) at each point on the lattice
        
        If the calculation has already been done, the result is simply returned and is not repeated.

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        particlesProducedDeriv : array(N, N)
        """
        if self._particlesProducedDerivExists and not forceCalculate:
            return self._particlesProducedDeriv

        # Make sure these quantities exist
        self.omegaFFT(verbose=verbose)
        self.momentaMagnitudeSquared(verbose=verbose) # This also calculates thetaInFourierSpace and momentaComponents

        if verbose > 0:
            print(f'Calculating {type(self).__name__} derivative of particles produced' + '.'*10, end='') 

        self._particlesProducedDeriv = _calculateParticlesProducedDerivOpt(self.N, self.gluonDOF, self._momentaMagSquared, self._omegaFFT)

        if verbose > 0:
            print('finished!')

        self._particlesProducedDerivExists = True

        return self._particlesProducedDeriv
 

    def particlesProduced(self, forceCalculate=False, verbose=0):
        r"""
        Compute the number of particles produced \(N(|k|)\) as a function of momentum. Note that this
        is technically the zeroth fourier harmonic, so this actually just calls the
        cgc.Collision.fourierHarmonic() function.

        The particles are binned according to cgc.Collision.momentaBins().
       
        Most likely will be plotted against cgc.Collision.momentaBins().

        If the calculation has already been done, the result is simply returned and is not repeated.


        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        particlesProduced : array(numBins = L / (delta 2 pi))
        """
        # This one is strictly real, so we should make sure that is updated        
        self._fourierHarmonics[0] = np.real(self.fourierHarmonic(0, forceCalculate, verbose))
        return self._fourierHarmonics[0]


    def fourierHarmonic(self, harmonic: int, forceCalculate=False, verbose=0):
        r"""
        Calculate the fourier harmonic of the particle production as:

        $$ v_n = \frac{ \sum_{(i,j)\in [k, k+ \Delta k]} |k| \frac{d^2 N}{d^2 k} e^{i n \theta }} { \sum_{(i,j)\in [k, k+ \Delta k]} |k| } $$

        If the calculation has already been done, the result is simply returned and is not repeated.


        Parameters
        ----------

        harmonic : int
            The fourier harmonic to calculate. All odd harmonics should be zero, and the zeroth harmonic
            will be equal to cgc.Collision.particlesProduced()

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        particlesProduced : array(numBins = L / (delta 2 pi))
        """
        # First, see if we have already calculated this harmonic
        if harmonic in self._fourierHarmonics.keys() and not forceCalculate:
            return self._fourierHarmonics[harmonic]

        # For actually calculating the harmonic, we first have to make sure we've calculated
        # the derivative, dN/d^2k
        # This makes sure that _momentaMagSquared, _thetaInFourierSpace and _particlesProducedDeriv
        # all exist
        self.particlesProducedDeriv(verbose=verbose)

        if verbose > 0:
            print(f'Calculating {type(self).__name__} fourier harmonic: {harmonic}' + '.'*10, end='') 

        # Drop all of our arrays into long 1D structure, since we will want to bin them
        vectorizedParticleDerivs = np.reshape(self._particlesProducedDeriv, [self.N*self.N])
        vectorizedTheta = np.reshape(self._thetaInFourierSpace, [self.N*self.N])
        vectorizedMomentaMag = np.reshape(np.sqrt(self._momentaMagSquared), [self.N*self.N])
       
        # The number of particles that are produced in each bin
        # These bins are actually just thin rings in momentum space
        self._fourierHarmonics[harmonic] = np.zeros(self.numBins, dtype='complex')
        # The bin sizes/bounds are calculated for elsewhere
        self.momentaBins()

        # Ideally, these rings should be only have a thickness dk (infinitesimal)
        # but since we have a discrete lattice, we weight the particles by their momentum
        # (which may slightly vary) and then properly normalize

        # Go through each bin and calculate (for all points in that bin):
        # 1. Sum over |k| * dN/d^2k * exp(i * harmonic * theta)
        # 2. Sum over |k|
        # 3. Divide 1./2.
        for i in range(self.numBins):
            # Find which places on the lattice fall into this particular momentum bin
            # Note the use of element-wise (or bitwise) and, "&"
            particleDerivsInRing = vectorizedParticleDerivs[(vectorizedMomentaMag < self.binSize*(i+1)) & (vectorizedMomentaMag > self.binSize*i)]
            momentaMagInRing = vectorizedMomentaMag[(vectorizedMomentaMag < self.binSize*(i+1)) & (vectorizedMomentaMag > self.binSize*i)]
            thetaInRing = vectorizedTheta[(vectorizedMomentaMag < self.binSize*(i+1)) & (vectorizedMomentaMag > self.binSize*i)]

            # Note that multiplication is done element-wise by default
            numeratorSum = np.sum(particleDerivsInRing * momentaMagInRing * np.exp(1.j * harmonic * thetaInRing))
           
            denominatorSum = np.sum(momentaMagInRing)

            self._fourierHarmonics[harmonic][i] = numeratorSum / denominatorSum


        if verbose > 0:
            print('finished!')

        return self._fourierHarmonics[harmonic]


# Using custom functions within other jitted functions can cause some issues,
# so we define the signatures explicitly for these two functions.
@numba.jit((numba.float64[:,:], numba.int64, numba.int64, numba.int64, numba.float64), nopython=True, cache=CACHE_OPTIMIZATIONS)
def _x_deriv(matrix, i, j, N, delta):
    return (matrix[i,(j+1)%N] - matrix[i,j-1]) / (2 * delta)

@numba.jit((numba.float64[:,:], numba.int64, numba.int64, numba.int64, numba.float64), nopython=True, cache=CACHE_OPTIMIZATIONS)
def _y_deriv(matrix, i, j, N, delta):
    return (matrix[(i+1)%N,j] - matrix[i-1,j]) / (2 * delta)

# Because of the same issue described above, we can't cache this function
# This function gives a warning because numba only experimentally supports
# treating functions as objects (the list derivs).
@numba.jit(nopython=True)
def _calculateOmegaOpt(N, gluonDOF, delta, incidentGaugeField, targetAdjointWilsonLine):
    """
    Calculate the field omega at each point on the lattice.

    If the field already exists, it is simply returned and no calculation is done.

    Returns
    -------
    numpy.array : shape=(2, 2, `colorCharges`**2 - 1, N, N)

    """

    # 2,2 is for the 2 dimensions, x and y
    omega = np.zeros((2, 2, gluonDOF, N, N), dtype='complex') # 2 is for two dimensions, x and y

    derivs = [_x_deriv, _y_deriv]

    for i in range(N):
        for j in range(N):
            for k in range(gluonDOF):
                for l in range(2): # 2 is number of dimensions
                    for n in range(2): # 2 is number of dimensions
                        omega[l,n,k,i,j] = np.sum(np.array([derivs[l](incidentGaugeField[m], i, j, N, delta) * derivs[n](targetAdjointWilsonLine[k, m], i, j, N, delta) for m in range(gluonDOF)]))

    return omega


@numba.jit(nopython=True, cache=CACHE_OPTIMIZATIONS)
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
            momentaComponents[i,j] = [2/delta * np.sin(np.pi*i/N) * np.sign(np.sin(2*np.pi*i/N)), 2/delta * np.sin(np.pi*j/N) * np.sign(np.sin(2*np.pi*j/N))]
            theta[i,j] = np.arctan2(momentaComponents[i,j,1], momentaComponents[i,j,0])

    return momentaComponents, theta

@numba.jit(nopython=True, cache=CACHE_OPTIMIZATIONS)
def _calculateParticlesProducedDerivOpt(N, gluonDOF, momentaMagSquared, omegaFFT):
    """
    Optimized (via numba) function to calculate dN/d^2k

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
