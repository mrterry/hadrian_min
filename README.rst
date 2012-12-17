hadrian_min
-----------
hadrian_min is a stochastic, hill climbing minimization algorithm.  It
uses a stratified sampling technique (Latin Hypercube) to get good
coverage of potential new points.  It also uses vectorized function
evaluations to drive concurrent function evaluations.

It is named after the Roman Emperor Hadrian, the most famous Latin hill
mountain climber of ancient times.

It is implemented in 2D only, but could be exteneded to higher/lower
dimensions if you workout the how to do the sampling in a more general way
or for your special case.
