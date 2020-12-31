import numpy as np

from Wavefunctions import Wavefunction

class Collision():

    targetWavefunction = None # Implements wilson line
    incidentWavefunction = None # Doesn't (have to) implement wilson line

    _omega = None
    _omegaFFT = None
    _particleProduction = None

    _omegaExists = False
    _omegaFFTExists = False
    _particleProductionExists = False

    def __init__(self, wavefunction1: Wavefunction, wavefunction2: Wavefunction):
        """
        Initialize a collision with two wavefunctions, presumably a nucleus and a proton. The first is expected to implement
        the wilson line, though as long as only one of the wavefunctions has this property, it will detect the proper one

        In the case that both wavefunctions implement the wilson line, the first (wavefunction1) will be used as such

        In the case that neither implement the wilson line, this method will return an error
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
        self.gluonDOF = self.targetWavefunction.gluonDOF
        self.delta = self.targetWavefunction.delta

        #print(self.targetWavefunction)
        #print(self.incidentWavefunction)


    def omega(self):

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
                            self._omega[l,n,k,i,j] = np.sum([derivs[l](self.incidentWavefunction.gaugeField()[m], i, j) * derivs[n](self.targetWavefunction.adjointWilsonLine()[k+1, m+1], i, j) for m in range(self.gluonDOF)])

        return self._omega
