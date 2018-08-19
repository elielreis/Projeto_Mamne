from numba import guvectorize
from numpy import shape


def ME_CST(xg1, xg2, xg3, yg1, yg2, yg3, rho, t):
    from numpy import array

    xg12 = xg1 - xg2
    xg31 = xg3 - xg1
    yg12 = yg1 - yg2
    yg31 = yg3 - yg1

    A = abs(xg31 * yg12 - xg12 * yg31) * 0.5
    me = (rho * t * A / 12.) * array([[2., 0., 1., 0., 1., 0.],
                                      [0., 2., 0., 1., 0., 1.],
                                      [1., 0., 2., 0., 1., 0.],
                                      [0., 1., 0., 2., 0., 1.],
                                      [1., 0., 1., 0., 2., 0.],
                                      [0., 1., 0., 1., 0., 2.]])

    return me


def VMG_CST(rho, t, coord, connec_face, arrayfalse, out):

    nglel = 6

    p, q = shape(coord)
    pp, qq = shape(connec_face)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x1, x2, x3 = coord[connec_face[i], 0]
        y1, y2, y3 = coord[connec_face[i], 1]

        me = ME_CST(x1, x2, x3, y1, y2, y3, rho, t)

        ve = me.reshape(me.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VMG_CST = guvectorize(['float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                                   'float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                                  '(),(),(p,q),(pp,qq),(k)->(k)')(VMG_CST)


def MG_CST(rho, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vmg = GU_VMG_CST(rho, t, coord, connec_face, arrayfalse)
    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    return mg