from numba import guvectorize
from numpy import shape, array
from numpy.linalg import det


def VME_TETRA(rho, volume):
    from numpy import array

    vme = ((rho * volume) / 20.) * \
        array([2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
               0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
               0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1,
               1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1,
               1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1,
               1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0,
               0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2], float)

    return vme


def VMG_TETRA(rho, coord, connec_volume, arrayfalse, out):

    nglel = 12

    p, q = shape(coord)
    pp, qq = shape(connec_volume)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):

        x0, x1, x2, x3 = coord[connec_volume[i], 0]
        y0, y1, y2, y3 = coord[connec_volume[i], 1]
        z0, z1, z2, z3 = coord[connec_volume[i], 2]

        volume = abs((1. / 6.) * det(array([[1., x0, y0, z0],
                                            [1., x1, y1, z1],
                                            [1., x2, y2, z2],
                                            [1., x3, y3, z3]])))

        ve = VME_TETRA(rho, volume)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VMG_TETRA = guvectorize(['float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                            'float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                           '(),(p,q),(pp,qq),(k)->(k)')(VMG_TETRA)


def MG_TETRA(rho, ngl, coord, connec_volume, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vmg = GU_VMG_TETRA(rho, coord, connec_volume, arrayfalse)
    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    return mg
