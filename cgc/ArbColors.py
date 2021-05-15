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

    def __init__(self, colorCharges, N, delta, mu, M=.5, g=1, Ny=100):
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

        """

        super().__init__(colorCharges, N, delta, mu, M, g) # Super constructor
        self._basis = get_basis(colorCharges)
        self.Ny = Ny

        # Modify the gaussian width to account for the multiple longitudinal layers
        self.gaussianWidth = self.mu / self.delta / np.sqrt(self.Ny)


    def colorChargeField(self):
        r"""
        Generates the color charge density field according to a gaussian distribution. Differs
        from super class implementation in that it generates the numerous fields according
        to `Ny`. That is, the field \(\rho\) satisfies:

        $$ \langle \rho_{a}^{(t)}(i^-,\vec i_{\perp}) \rho_{b}^{(t)}({j^-},\vec j_{\perp}) \rangle = g^2\mu_t^2 \frac{ 1 }{N_y \Delta^2}  ~\delta_{ab}~\delta_{i_{\perp,1}\ j_{\perp,1}}~\delta_{i_{\perp,2} \ j_{\perp,2}} ~\delta_{i^- \ {j^-}} $$ 


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
        r"""
        Calculates the gauge field for all longitudinal layers and charge distributions by solving the (modified)
        Poisson equation involving the color charge field

        $$g \frac{1  } {\partial_\perp^2 - m^2 } \rho_a(i^-, \vec {i}_\perp )$$

        via Fourier method.

        If the field already exists, it is simply returned and no calculation is done.
        """

        if self._gaugeFieldExists:
            return self._gaugeField

        # Make sure the charge field has already been generated (if not, this will generate it)
        self.colorChargeField()

        # Compute the fourier transform of the charge field
        # Note that the normalization is set to 'backward', which for scipy means that the
        # ifft2 is scaled by 1/n (where n = N^2)
        chargeDensityFFTArr = fft2(self._colorChargeField, axes=(-2,-1), norm='backward')

        # Absorb the numerator constants in the equation above into the charge density
        chargeDensityFFTArr = -self.delta2 * self.g / 2 * chargeDensityFFTArr

        # Calculate the individual elements of the gauge field in fourier space
        # Note here that we have to solve the gauge field for each layer and for each gluon degree of freedom
        # This method is defined at the bottom of this file; see there for more information
        gaugeFieldFFTArr = _calculateGaugeFFTOpt(self.gluonDOF, self.N, self.Ny, self.poissonReg, chargeDensityFFTArr); 

        # Take the inverse fourier transform to get the actual gauge field
        self._gaugeField = np.real(ifft2(gaugeFieldFFTArr, axes=(-2, -1), norm='backward'))

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
        self.gaugeField()

        # We now combine all of the longitudinal layers into the single wilson line
        # Optimized method is defined at the end of this file; see there for more information
        self._wilsonLine = _calculateWilsonLineOpt(self.N, self.Ny, self.colorCharges, self._basis, self._gaugeField)
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

        # Calculation is optimized with numba, as with the previous calculations
        # See bottom of the file for more information
        self._adjointWilsonLine = _calculateAdjointWilsonLineOpt(self.gluonDOF, self.N, self._basis, self._wilsonLine)
        self._adjointWilsonLineExists = True

        return self._adjointWilsonLine



# Since we want to speed up the calculate, we define the calculation of the fourier elements of
# the gauge field using a numba-compiled method
# This has to be defined outside of the Nuclelus class since numbda doesn't play well with custom libraries
@numba.jit(nopython=True)
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
                    gaugeFieldFFTArr[l,k,i,j] = chargeDensityFFTArr[l,k,i,j]/(2 - np.cos(two_pi_over_N*i) - np.cos(two_pi_over_N*j) + poissonReg) 
    return gaugeFieldFFTArr


# Same deal as the above method, we have to define it outside the class so
# numba doesn't get confused
@numba.jit(nopython=True) 
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
                    expArgument = expArgument + gaugeField[l,k,i,j] * basis[k]
                
                # Now actually calculate the exponential with our custom defined expm method
                # that can properly interface with numba (scipy's can't)
                exponential = np.ascontiguousarray(expm(-1.j * expArgument))
                wilsonLine[i,j] = np.dot(wilsonLine[i,j], exponential)

    return wilsonLine


@numba.jit(nopython=True)
def _calculateAdjointWilsonLineOpt(gluonDOF, N, basis, wilsonLine):
    r"""
    Calculate the wilson line in the adjoint representation.

    This method is optimized using numba.
    """
    # Wilson line is always real in adjoint representation, so need to dtype='complex' as with the others
    adjointWilsonLine = np.zeros((gluonDOF, gluonDOF, N, N), dtype='double')

    for a in range(gluonDOF):
        for b in range(gluonDOF):
            for i in range(N):
                for j in range(N):
                    V = wilsonLine[i,j]
                    Vdag = np.conjugate(np.transpose(V))
                    adjointWilsonLine[a,b,i,j] = 2 * np.real(np.trace(np.dot(np.dot(basis[a], V), np.dot(basis[b], Vdag))))

    return adjointWilsonLine
