# Color Glass Condensate Simulations

This is a toolbox to model color glass condensates.

## Documentation

You can access the documentation for the project [here](https://jfeatherstone.github.io/ParticleProduction/docs/index.html).

## Optimization

More work needs to be done on the optimization of individual methods, but most calculations are done with efficiency in mind.

All calculation methods are properly chained, which means the following two pieces of code perform the same number of calculations:

```
nucleus = Nucleus(2, N, delta, mu)
proton = Proton(2, N, delta, mu, radius)
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
nucleus = Nucleus(2, N, delta, mu)
proton = Proton(2, N, delta, mu, radius)
col = Collision(nucleus, proton)

plt.plot(col.momentaBins(), col.particlesProduced())
```
