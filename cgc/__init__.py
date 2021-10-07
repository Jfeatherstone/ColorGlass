
r"""

This toolbox offers the funtionality to simulate color glass condensates in the dilute-dense approximation.

## **Overview of Classes**

### `Wavefunction`

Defines the base methods in what will later be the super class for both the proton and nucleus objects.
This class implements methods for generating a random color charge density according to a guassian
distribution, as well as solving the laplace equation for these densities to get the gauge field.

The `Wavefunction.Proton` class is defined here because it only trivially depends on the number of possible color
charges, whereas the nuclei class has a more involved dependence (see below).

### `ArbColors`

Defines the nucleus object for an arbitrary number of colors, with the exponential calcuation being
done as it is with three colors above. The basis matrices are generated as the generalized
Gell-Mann matrices.

### `Collision`

Defines a collision object that takes in two `Wavefunction.Wavefunction` and calculates the properties of their collision.
At least one of the parameters must be a `Wavefunction.Wavefunction` that implements a calculation of the Wilson Line
(eg. `ArbColors.Nucleus`)

### **Deprecated Classes**

As a test of the toolbox functionality, the \(SU(2)\) and \(SU(3)\) cases have been explicitly defined, where the former
is done using a closed form the exponential of a matrix. The latter is simply a less generalized version of the
arbitrary color case above, in that it uses a numerical form for diagonalizing and computing the exponential
of a matrix when calculating the Wilson Line.

In practical applications, it is easier to simply use the `ArbColors.Nucleus` and `Wavefunction.Proton` classes for everything,
though the classes below are able to interface with the ones above.

### `TwoColors`

Defines the nucleus object for two possible color charges, where the calculation of the Wilson Line
can be done using the closed form of the exponential of the Pauli matrices.

### `ThreeColors`

Defines the nucleus object for three possible color charges, where the calculation of the Wilson Line
must be done using a numerical exponentiation method.


## **Usage Examples**

The simplest usage example would be to create nucleus and proton objects to calculate the momenta of particles produced:

First, import the toolbox and whatever other packages you will use for post-processing

    import cgc
    import matplotlib.pyplot as plt

Define constants for the dilute and dense objects.

    N = 128         # Number of lattice sites
    delta = .1      # Inter-lattice spacing
    mu = 2          # Gaussian distribution width for color field generation
    radius = 1      # Radius of dilute proton

Create the dilute and dense objects for \(SU(3)\), as well as a collision between them. Note that no actual calculations are done in this step;
rather, calculations are only performed as they are deemed necessary (ie. when a particular quantity is called)

    nucelus = cgc.Nucleus(3, N, delta, mu)
    proton = cgc.Proton(3, N, delta, mu, radius)

    col = cgc.Collision(nucleus, proton)

From here, quantities can be generated from calling any of the three objects above.

    plt.plot(col.momentaBins(), col.particlesProduced())

<details>
<summary>Full Code</summary>
<p>

```
    import cgc
    import matplotlib.pyplot as plt

    N = 128         # Number of lattice sites
    delta = .1      # Inter-lattice spacing
    mu = 2          # Gaussian distribution width for color field generation
    radius = 1      # Radius of dilute proton

    nucelus = cgc.Nucleus(3, N, delta, mu)
    proton = cgc.Proton(3, N, delta, mu, radius)

    col = cgc.Collision(nucleus, proton)

    plt.plot(col.momentaBins(), col.particlesProduced())
```

</p>
</details>

"""

VERSION = "1.0"

from .Wavefunction import Wavefunction, Proton
from .Collision import Collision

# No longer include the explicit SU(n) representations, since the arbitrary one
# works very well
#from .TwoColors import Nucleus as Nucleus2
#from .ThreeColors import Nucleus as Nucleus3
from .ArbColors import Nucleus as Nucleus
