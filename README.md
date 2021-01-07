# Particle Production in Relativistic Collisions

This is a toolbox to calculate the particles, grouped by their momenta, that are created during relativistic gluon interactions involving a nucleus and a proton.

## Class structure

Currently, the generic class that is then extended for the objects involved in the collision is `Wavefunction`. This class implements just a basic color charge density and gauge field. This class can be implemented for any number of gluon degrees of freedom, meaning it will work for either 2 or 3 color charges.

The only part of the calculation that greatly changes upon using 2 vs. 3 color charges is the calculation of the Wilson line, so this method is unimplemented here.

For a two color system, the `TwoColors.py` file defines a `Nucleus` and `Proton`, where the former implements the Wilson line calculations, while the latter reimplements the color charge generation to weight it by a centered Gaussian field.

A three color system equivalent is in the process of being implemented.

Finally, a `Collision` can be created from two `Wavefunction` (or child) classes, regardless of the number of color charges. At least one of the classes passed to this constructor must implement the `wilsonLine()` method, otherwise an error will be thrown.

## Attributes that can be calculated for each object:

### Wavefunction (generic)

- colorChargeField()
- gaugeField()

### Nucleus

- All Wavefunction attributes
- wilsonLine()
- adjointWilsonLine()

### Proton

- All Wavefunction attributes


### Collision

- omega()
- omegaFFT()
- momentaBins()
- particlesProduced()

All calculation methods are properly chained, which means the following two pieces of code perform the same number of calculations:

```
nucleus = Nucleus(N, delta, mu)
proton = Proton(N, delta, mu, radius)
col = Collision(nucleus, proton)

nucleus.colorChargeField()
nucleus.gaugeField()
nucleus.wilsonLine()
nucleus.adjointWilsonLine()

proton.colorChargeField()
proton.gaugeField()

col.omega()
col.omegaFFT()
col.momentaBins()
col.particlesProduced()

plt.plot(col.momentaBins(), col.particlesProduced())
```

```
nucleus = Nucleus(N, delta, mu)
proton = Proton(N, delta, mu, radius)
col = Collision(nucleus, proton)

plt.plot(col.momentaBins(), col.particlesProduced())
```
