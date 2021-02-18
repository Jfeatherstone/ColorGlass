
r"""

This toolbox offers methods to calculate the number of particles and their momenta that are produced
in color glass condensates.

toolbox.Wavefunction
--------------------

Defines the base methods in what will later be the super class for both the proton and nucleus objects.
This class implements methods for generating a random color charge density according to a guassian
distribution, as well as solving the laplace equation for these densities to get the gauge field.

The `Proton` class is defined here because it only trivially depends on the number of possible color
charges, which the nuclei classes for various special unitary groups are defined separately (see below).

toolbox.TwoColors
-----------------

Defines the nucleus object for two possible color charges, where the calculation of the Wilson Line
can be done using the closed form of the exponential of the Pauli matrices.

toolbox.ThreeColors
-------------------

Defines the nucleus object for three possible color charges, where the calculation of the Wilson Line
must be done using a numerical exponentiation method.

toolbox.ArbColors
-----------------

Defines the nucleus object for an arbitrary number of colors, with the exponential calcuation being
done as it is with three colors above. The basis matrices are generated as the generalized
Gell-Mann matrices.

toolbox.Collision
-----------------

Defines a collision object that takes in two `Wavefunctions` and calculates the properties of their collision.
At least one of the parameters must be a `Wavefunction` that implements a calculation of the Wilson Line.

"""

from .Wavefunction import Wavefunction, Proton
from .Collision import Collision

from .TwoColors import Nucleus as Nucleus2
from .ThreeColors import Nucleus as Nucleus3
from .ArbColors import Nucleus as Nucleus
