# Color Glass Condensate Simulations

This is a toolbox to model color glass condensates in the dense-dilute approximation.

## Documentation

You can access the documentation for the project [here](https://jfeatherstone.github.io/ColorGlass/).

## Usage Examples

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

## Optimization

Computationally expensive calculations (nucleus gauge field, wilson line, and adjoint wilson line) are optimized using the `numba` library, which transpiles the functions into C code.

## References

(not complete list yet)

Higham, N. J. (2008). _Functions of matrices: Theory and computation_. Society for Industrial and Applied Mathematics.

Bertlmann, R. A., & Krammer, P. (2008). Bloch vectors for qudits. Journal of Physics A: Mathematical and Theoretical, 41(23), 235303. [10.1088/1751-8113/41/23/235303](https://doi.org/10.1088/1751-8113/41/23/235303)
