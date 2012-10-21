import numpy as np
from scipy.stats import uniform
from scipy.spatial import Delaunay
from itertools import product


def latin_hypercube_sample(distributions, npoints):
    """
    Sample ``npoints`` from a sequences of scipy.stats distributions.  Using a
    Latin Hypercube stratified sampling method.

    Parameters
    ----------
    distributions: list of scipy.stats distributions
        objects with a ppf (inverse CDF) function for sampling the
        distribution

    npoints: int
        number of points to sample / subdivisions for each dimension
    
    Returns
    -------
    samples: array shape=(npoints, len(distributions))
        sequence of ``npoints`` points sampled from distributions
    
    Examples
    --------
    >>> dist = scipy.stats.norm(loc=20, scale=0.3)
    >>> pts = latin_hypercube_sample([dist], 30)
    """
    ndims = len(distributions)
    assert npoints > np.ceil(4. * ndims / 3.), "npoints is too small"
    samples = np.empty((npoints, ndims))

    # break each dimension into (npoints) equal probability chunks
    # sample from each chunk and shuffle
    percentiles = np.linspace(0, 1, npoints + 1)
    for d, dist in enumerate(distributions):
        cdf_pts = np.random.uniform(percentiles[:-1], percentiles[1:])
        samples[:, d] = dist.ppf(cdf_pts)
        np.random.shuffle(samples[:, d])
    
    return samples


def timing_slope(pts, t0=16.e-3, m0=25.):
    return (np.power((pts[:, 0]-t0)/t0, 2) + np.power((pts[:, 1]-m0)/m0, 2))


def sample_triangle(points):
    """
    Sample 1 point from a given triangle
    """
    npoints, nsides, ndims = points.shape
    assert ndims == 2
    assert nsides == 3

    r12 = np.random.random((npoints, 2))
    r1 = np.sqrt(r12[:, 0])
    r2 = r12[:, 1]

    r = np.zeros((npoints, 3))
    r[:, 0] = 1-r1
    r[:, 1] = r1 * (1-r2)
    r[:, 2] = r1 * r2

    pts = np.zeros((npoints, 2))
    pts = (points * r[:, :, None]).sum(1)
    return pts


def area(pts):
    ans = ( pts[:, 0, 0]*(pts[:, 1, 1]-pts[:, 2, 1]) +
            pts[:, 1, 0]*(pts[:, 2, 1]-pts[:, 0, 1]) +
            pts[:, 2, 0]*(pts[:, 0, 1]-pts[:, 1, 1])
            )
    return abs(ans)/2.


@profile
def random_min(para_eval, xbnds, ybnds, xtol, ytol, swarm=8, mx_iters=5):
    assert xbnds[1] > xbnds[0]
    assert ybnds[1] > ybnds[0]
    assert xtol > 0
    assert ytol > 0
    # simplexes are simplex indexes
    # vertexes are vertex indexes
    # points are spatial coordinates

    samples = [
            uniform(xbnds[0], xbnds[1]-xbnds[0]),
            uniform(ybnds[0], ybnds[1]-ybnds[0]),
            ]
    points = latin_hypercube_sample(samples, swarm)
    z = para_eval(points)

    # exclude corners from possibilities, but add them to the triangulation
    # this bounds the domain, but ensures they don't get picked
    points = np.append(points, list(product(xbnds, ybnds)), axis=0)
    z = np.append(z, 4*[z.max()])

    tri = Delaunay(points)
    del points

    for step in range(1, mx_iters+1):
        i, vertexes = get_vertexes(tri, z, n_tries=4) 
        disp = tri.points[i] - tri.points[np.unique(vertexes)]
        disp /= np.array([xtol, ytol])
        err = err_mean(disp)

        if err < 1.:
            return tri.points[i], z[i], tri.points, z, 1

        tri_points = tri.points[vertexes]
        bnds = np.cumsum(area(tri_points))
        bnds /= bnds[-1]
        indx = np.searchsorted(bnds, np.random.rand(swarm))

        new_pts = sample_triangle(tri_points[indx])
        new_z = para_eval(new_pts)

        points = np.append(tri.points, new_pts, axis=0)
        tri = Delaunay(points)
        z = np.append(z, new_z)

    return None, None, tri.points, z, step


def get_vertexes(tri, z, n_tries=4):
    """
    Sometimes a point with not be in a scipy.spatial.Delaunay triangulation.
    Find vertex with minimum value that is actually in the triangulation.
    """
    i = z.argmin()
    simplexes = (tri.vertices == i).any(1)
    for t in range(n_tries):
        if any(simplexes):
            vertexes = tri.vertices[simplexes]
            return i, vertexes

        i1 = z[:i].argmin()
        try:
            i2 = z[i+1:].argmin()
            if z[i1] < z[i+i2]:
                i = i1
            else:
                i = i + i2
        except:
            i = i1
        simplexes = (tri.vertices == i).any(1)
    raise "can't find vertex in triangulation"


def err_mean(disp):
    assert len(disp) > 0
    disp = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)
    return disp.max()


def main():
    np.random.seed(1)

    for tmp in range(100):
        xbnds = (10.e-3, 25.e-3)
        ybnds = (10, 45)
        pt, z, all_pts, all_vals, step = random_min(timing_slope,
                xbnds=xbnds, ybnds=ybnds,
                xtol=50.e-6, ytol=5.,
                mx_iters=25,
                )

main()