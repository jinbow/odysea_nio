# Analytical tools for the Odysea science

The codes for Ocean Dynamics and Surface Exchange with the Atmosphere (Odysea) development.

## Near Inertial Oscillation reconstruction

The NIO is reconstructed through optimizing a slab model prediction constrainted by the Odysea observations. 

The Slab model follows 
$$ \frac{du}{dt}-(f+ \zeta/2)v&=&\frac{\tau_x}{H \rho} -ru$$
$$ \frac{dv}{dt}+(f+ \zeta/2)u&=&\frac{\tau_y }{H\rho}  -rv$$

where $H$ represents mixed layer depth, $f$ the inertial frequency, $\zeta$ the relative vorticity associated with mesoscale eddies and submesoscale structures, $\tau_x$ the zonal wind stress, $\tau_y$ the meridional wind stress, $\rho$ mixed layer density, $r$ the damping coefficient, and $u$ and $v$ the zonal and meridional velocities, respectively. The frequency shift due to $\zeta$ is explained in Kunze (1985).

For a forecast model, this linear slab model can predict NIO give an initial velocity, mixed layer depth, wind stress vector and damping coefficient. We write this forecast model as 
$$ \overrightarrow{u^f}=\mathcal{F}(\overrightarrow{u}_0, \overrightarrow{\tau} , H, r)$$

The hypothesis is that given the infrequent and possibly aliased wind stress and surface currents from WaCM, we can utilize the predictive power of the slab model constrained by the WaCM observations to reconstruct the NIO. This is an under-determined problem that relies on optimizations for the best solution.  We take the difference between the forecast slab model velocities and the WaCM observed velocities as the cost function of the optimization
$$    J(\overrightarrow{u}_0,  H, r) = \| \overrightarrow{u^f}(t_{w}) - \overrightarrow{u}_{w} \| $$ where $t_w$ is the time of the WaCM observations and $\overrightarrow{u^f}(t_{w})$ is the slab model solution at $t_w$. An optimal set of control parameters $(\overrightarrow{u}_0,  H, r)$ is found by minimizing $J$. 
