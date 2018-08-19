from numba import guvectorize
from numpy import shape


def ME_DKT(xg1, xg2, xg3, yg1, yg2, yg3, rho, t):
    from numpy import eye

    nglel = 9

    xg12 = xg1 - xg2
    xg31 = xg3 - xg1
    yg12 = yg1 - yg2
    yg31 = yg3 - yg1

    A = abs(xg31 * yg12 - xg12 * yg31) * 0.5

    me_lumped = rho * (1. / 3.) * A * t * eye(nglel)
    me_lumped[1, 1] = 0.
    me_lumped[2, 2] = 0.
    me_lumped[4, 4] = 0.
    me_lumped[5, 5] = 0.
    me_lumped[7, 7] = 0.
    me_lumped[8, 8] = 0.

    return me_lumped


def VMG_DKT(rho, t, coord, connec_face, arrayfalse, out):
    
    nglel = 9

    p, q = shape(coord)
    pp, qq = shape(connec_face)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x1, x2, x3 = coord[connec_face[i], 0]
        y1, y2, y3 = coord[connec_face[i], 1]

        me = ME_DKT(x1, x2, x3, y1, y2, y3, rho, t)

        ve = me.reshape(me.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VMG_DKT = guvectorize(['float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                          'float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                         '(),(),(p,q),(pp,qq),(k)->(k)')(VMG_DKT)


def MG_DKT(rho, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vmg = GU_VMG_DKT(rho, t, coord, connec_face, arrayfalse)
    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    return mg
