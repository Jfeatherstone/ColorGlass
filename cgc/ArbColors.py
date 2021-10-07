from .Wavefunction import Wavefunction
from .LinAlg import expm, get_basis

import numba

import numpy as np
from scipy.fft import ifft2, fft2

class Nucleus(Wavefunction):

    # Upon calling wilsonLine() or adjointWilsonLine(), these are properly defined
    _wilsonLine = None
    _adjointWilsonLine = None

    # Some variables to keep track of what has been calculated/generated so far
    # allowing us to avoid redundant computations
    _wilsonLineExists = False
    _adjointWilsonLineExists = False

    def __init__(self, colorCharges, N, delta, mu, M=.5, g=1, Ny=100, rngSeed=None):
        r"""
        Dense object to be used in an instance of `cgc.Collision.Collision`.

        Implements calculation of the Wilson Line using the generalized basis matrix set.

        Parameters
        ----------

        colorCharges : positive integer
            The number of possible color charges; also the dimensionality of the special unitary group.

        N : positive integer
            The size of the square lattice to simulate.

        delta : positive float
            The distance between adjacent lattice sites.

        mu : positive float
            The scaling for the random gaussian distribution that generates the color charge density.

        M : float (default=.5)
            Infrared regulator parameter to regularize the Poisson equation for the gauge field.

        g : float (default=1)
            Parameter in the Poisson equation for the gauge field.

        Ny : positive integer (default=100)
            The longitudinal extent (in layers) of the nucleus object.

        rngSeed : int (default=None)
            Seed for the random number generator to initialize the color charge field
        """

        super().__init__(colorCharges, N, delta, mu, M, g, rngSeed) # Super constructor
        self._basis = get_basis(colorCharges)
        self.Ny = Ny

        # Modify the gaussian width to account for the multiple longitudinal layers
        self.gaussianWidth = self.mu / self.delta / np.sqrt(self.Ny)


    def colorChargeField(self, forceCalculate=False, verbose=0):
        r"""
        Generates the color charge density field according to a gaussian distribution. Differs
        from super class implementation in that it generates the numerous fields according
        to `Ny`. That is, the field \(\rho\) satisfies:

        $$ \langle \rho_{a}^{(t)}(i^-,\vec i_{\perp}) \rho_{b}^{(t)}({j^-},\vec j_{\perp}) \rangle = g^2\mu_t^2 \frac{ 1 }{N_y \Delta^2}  ~\delta_{ab}~\delta_{i_{\perp,1}\ j_{\perp,1}}~\delta_{i_{\perp,2} \ j_{\perp,2}} ~\delta_{i^- \ {j^-}} $$ 

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

        colorChargeField : array(Ny, N, N, `colorCharges`**2 - 1)
        """
        if self._colorChargeFieldExists and not forceCalculate:
            return self._colorChargeField

        if verbose > 0:
            print(f'Generating {type(self).__name__} color charge field' + '.'*10, end='')

        # To compare to old results
        #self._colorChargeField = self.rng.normal(scale=self.gaussianWidth, size=(self.Ny, self.gluonDOF, self.N, self.N))
        #self._colorChargeField = self._colorChargeField.swapaxes(1, 2)
        #self._colorChargeField = self._colorChargeField.swapaxes(2, 3)

        # Randomly generate the intial color charge density using a gaussian distribution
        self._colorChargeField = self.rng.normal(scale=self.gaussianWidth, size=(self.Ny, self.N, self.N, self.gluonDOF))
        # Make sure we don't regenerate this field since it already exists on future calls
        self._colorChargeFieldExists = True

        if verbose > 0:
            print('finished!')

        return self._colorChargeField


    def gaugeField(self, forceCalculate=False, verbose=0):
        r"""
        Calculates the gauge field for all longitudinal layers and charge distributions by solving the (modified)
        Poisson equation involving the color charge field

        $$g \frac{1  } {\partial_\perp^2 - m^2 } \rho_a(i^-, \vec {i}_\perp )$$

        via Fourier method.

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

        gaugeField : array(Ny, N, N, `colorCharges`**2 - 1)
        """

        if self._gaugeFieldExists and not forceCalculate:
            return self._gaugeField

        # Make sure the charge field has already been generated (if not, this will generate it)
        self.colorChargeField(verbose=verbose)

        if verbose > 0:
            print(f'Calculating {type(self).__name__} gauge field' + '.'*10, end='')

        # Compute the fourier transform of the charge field
        # Note that the normalization is set to 'backward', which for scipy means that the
        # ifft2 is scaled by 1/n (where n = N^2)
        chargeDensityFFTArr = fft2(self._colorChargeField, axes=(1,2), norm='backward')

        # Absorb the numerator constants in the equation above into the charge density
        chargeDensityFFTArr = -self.delta2 * self.g / 2 * chargeDensityFFTArr

        # Calculate the individual elements of the gauge field in fourier space
        # Note here that we have to solve the gauge field for each layer and for each gluon degree of freedom
        # This method is defined at the bottom of this file; see there for more information
        gaugeFieldFFTArr = _calculateGaugeFFTOpt(self.gluonDOF, self.N, self.Ny, self.poissonReg, chargeDensityFFTArr); 

        # Take the inverse fourier transform to get the actual gauge field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(1,2), norm='backward'))

        # Make sure this process isn't repeated unnecessarily by denoting that it has been done
        self._gaugeFieldExists = True

        if verbose > 0:
            print('finished!')

        return self._gaugeField


    def wilsonLine(self, forceCalculate=False, verbose=0):
        """
        Calculate the Wilson line using the gauge field and the appropriate basis matrices.

        If the line already exists, it is simply returned and no calculation is done.
        

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        wilsonLine : array(N, N, `colorCharges`)
        """
        if self._wilsonLineExists and not forceCalculate:
            return self._wilsonLine

        # Make sure the gauge field has already been calculated
        self.gaugeField(verbose=verbose)

        if verbose > 0:
            print(f'Calculating {type(self).__name__} wilson line' + '.'*10, end='')

        # We now combine all of the longitudinal layers into the single wilson line
        # Optimized method is defined at the end of this file; see there for more information
        self._wilsonLine = _calculateWilsonLineOpt(self.N, self.Ny, self.colorCharges, self._basis, self._gaugeField)
        self._wilsonLineExists = True

        if verbose > 0:
            print('finished!')

        return self._wilsonLine

    def adjointWilsonLine(self, forceCalculate=False, verbose=0):
        """
        Calculate the Wilson line in the adjoint representation.

        If the line already exists, it is simply returned and no calculation is done.
        

        Parameters
        ----------

        forceCalculate : bool (default=False)
            If the quantity has previously been calculated, the calculation will not be done
            again unless this argument is set to True.

        verbose : int (default=0)
            How much output should be printed as calculations are done. Options are 0, 1, or 2.

        Returns
        -------

        adjointWilsonLine : array(N, N, `colorCharges`**2 - 1, `colorCharges`**2 - 1)
        """
        if self._adjointWilsonLineExists and not forceCalculate:
            return self._adjointWilsonLine
       
        # Make sure the wilson line has already been calculated
        self.wilsonLine(verbose=verbose)
 
        if verbose > 0:
            print(f'Calculating {type(self).__name__} adjoint wilson line' + '.'*10, end='')

        # Calculation is optimized with numba, as with the previous calculations
        # See bottom of the file for more information
        self._adjointWilsonLine = _calculateAdjointWilsonLineOpt(self.gluonDOF, self.N, self._basis, self._wilsonLine)
        self._adjointWilsonLineExists = True

        if verbose > 0:
            print('finished!')

        return self._adjointWilsonLine



# Since we want to speed up the calculate, we define the calculation of the fourier elements of
# the gauge field using a numba-compiled method
# This has to be defined outside of the Nuclelus class since numbda doesn't play well with custom classes
@numba.jit(nopython=True, cache=True)
def _calculateGaugeFFTOpt(gluonDOF, N, Ny, poissonReg, chargeDensityFFTArr):
    r"""
    Calculate the elements of the gauge field in fourier space.

    This method is optimized using numba.
    """
    gaugeFieldFFTArr = np.zeros_like(chargeDensityFFTArr, dtype='complex')

    # Precompute for speed
    two_pi_over_N = 2 * np.pi / N

    for l in range(Ny):
        for k in range(gluonDOF):
            for i in range(N):
                for j in range(N):
                    gaugeFieldFFTArr[l,i,j,k] = chargeDensityFFTArr[l,i,j,k]/(2 - np.cos(two_pi_over_N*i) - np.cos(two_pi_over_N*j) + poissonReg) 
    return gaugeFieldFFTArr


# Same deal as the above method, we have to define it outside the class so
# numba doesn't get confused
@numba.jit(nopython=True, cache=True) 
def _calculateWilsonLineOpt(N, Ny, colorCharges, basis, gaugeField): 
    r"""
    Calculate the elements of the wilson line.

    This method is optimized using numba.
    """
    wilsonLine = np.zeros((N, N, colorCharges, colorCharges), dtype='complex')       
    gluonDOF = colorCharges**2 - 1

    # Slightly different ordering of indexing than in other places in the code,
    # due to the fact that we have to sum of the gluonDOF and Ny axis
    for i in range(N):
        for j in range(N):

            # Create the unit matrix for each point since we are multiplying
            # the wilson line as we go (so we need to start with the identity)
            for c in range(colorCharges): 
                wilsonLine[i,j,c,c] = 1
            
            # The path ordered exponential becomes a product of exponentials for each layer
            for l in range(Ny):
                # Evaluate the argument of the exponential first 
                # We multiply the elements of the gauge field for each gluon degree of freedom
                # by the respective basis matrix and sum them together
                expArgument = np.zeros((colorCharges, colorCharges), dtype='complex') # Same shape as basis matrices
                for k in range(gluonDOF):
                    expArgument = expArgument + gaugeField[l,i,j,k] * basis[k]
                
                # Now actually calculate the exponential with our custom defined expm method
                # that can properly interface with numba (scipy's can't)
                exponential = np.ascontiguousarray(expm(-1.j * expArgument))
                wilsonLine[i,j] = np.dot(wilsonLine[i,j], exponential)

    return wilsonLine


@numba.jit(nopython=True, cache=True)
def _calculateAdjointWilsonLineOpt(gluonDOF, N, basis, wilsonLine):
    r"""
    Calculate the wilson line in the adjoint representation.

    This method is optimized using numba
    """
    # Wilson line is always real in adjoint representation, so need to dtype='complex' as with the others
    adjointWilsonLine = np.zeros((N, N, gluonDOF, gluonDOF), dtype='double')

    for a in range(gluonDOF):
        for b in range(gluonDOF):
            for i in range(N):
                for j in range(N):
                    V = wilsonLine[i,j]
                    Vdag = np.conjugate(np.transpose(V))
                    adjointWilsonLine[i,j,a,b] = 2 * np.real(np.trace(np.dot(np.dot(basis[a], V), np.dot(basis[b], Vdag))))

    return adjointWilsonLine
