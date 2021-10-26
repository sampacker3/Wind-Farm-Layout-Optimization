# windfarmp
Code that optimises the layout and number of wind turbines in offshore wind farms.

As the world’s power mix increasingly shifts towards renewable energy sources, the mathematically
complex multivariate nonlinear problem of wind farm layout optimisation is of important focus to
improve the cost effectiveness of their installation. The tool created in this project optimises
the layout and number of model specific turbines in an offshore wind farm for any macro-siting location
that wind and bathymetry data is provided for, and for any budget, such that the ‘cost per unit of energy’
objective function is minimized.

The tool models the costs associated with changing sea depth, fixed costs, turbine and maintenance costs,
and the costs associated with realistic cabling network constraints as input parameters. Energy output is
modelled as a function of wind speed using a turbine power curve, and the wind shadowing effects caused by
turbines drawing energy from the wind. The latter is modelled using a product of a Gaussian decay and a cosine
squared half wave angular dependence, with a range parameter, an angle parameter, and a scaling factor, in
replacement of a computationally expensive CFD model to reduce computation time. The wind shadowing model
could be replaced by a CFD model and the optimisation component would still function.

A variety of mathematical processes were used to create the tool, including but not limited to: optimisation,
graph theory surrounding capacited minimum spanning trees, and an adaption of k-means clustering.

The key limitation of the developed tool is the inability to sufficiently model the electrical constraints
that determine the turbine capacity of a cable that connects to the wind farm’s substation. It can thus be found
that an arbitrarily placed capacity limitation prohibits the ability to spread turbines apart in a cost-effective
manner that sufficiently mitigates the increased wind shadowing effect that an increase in turbines incurs.
It can therefore be concluded that including cable capacity as an optimisable parameter would be a worthwhile addition.

A list of dependencies and further instructions for use will be added soon. In the meantime, feel free to contact me at:
hickman.aaron07@gmail.com for any further enquires. Main code base is within the file Base_Costs_Code.ipynb.

This project was my first attempt at using Python, so any feedback sent to the email above would also be appreciated :)
