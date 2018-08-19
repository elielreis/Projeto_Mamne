from numba import guvectorize
from numpy import shape


def ME_CASCA(x1, x2, x3, y1, y2, y3, z1, z2, z3, rho, t):
    from FUNCTIONS_CASCA import LAMB
    from numpy import array, eye


    '''
        PARAMETERS: xi, yi: float64
                            Global Cartesian coordinates in
                            3d Euclidean space.
                    E:  float64
                        Modulus of linear elasticity.
                    v:  float64
                        Poisson Coefficient.
                    t:  float64
                        Thickness
        RETURNS:    out:    ndarray.dtype(float64)
                            Element stiffness matrix FLAT in global coordinates
                            of the plane of the element surface.
    '''

    lamb = LAMB(x1, x2, x3, y1, y2, y3, z1, z2, z3)

    coord_global = array([[x1, x2, x3],
                          [y1, y2, y3],
                          [z1, z2, z3]], float)

    coord_local = lamb.T.dot(coord_global)

    x1l, x2l, x3l = coord_local[0]
    y1l, y2l, y3l = coord_local[1]

    xl12 = x1l - x2l
    xl31 = x3l - x1l
    yl12 = y1l - y2l
    yl31 = y3l - y1l

    A = abs(xl31 * yl12 - xl12 * yl31) * 0.5

    me_lumped = rho * (1. / 3.) * A * t * eye(18, dtype=float)
    me_lumped[3, 3] = 0.
    me_lumped[4, 4] = 0.
    me_lumped[5, 5] = 0.
    me_lumped[9, 9] = 0.
    me_lumped[10, 10] = 0.
    me_lumped[11, 11] = 0.
    me_lumped[15, 15] = 0.
    me_lumped[16, 16] = 0.
    me_lumped[17, 17] = 0.

    return me_lumped


def VMG_CASCA(rho, t, coord, connec_face, arrayfalse, out):

    nglel = 18

    p, q = shape(coord)
    pp, qq = shape(connec_face)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x1, x2, x3 = coord[connec_face[i], 0]
        y1, y2, y3 = coord[connec_face[i], 1]
        z1, z2, z3 = coord[connec_face[i], 2]

        me = ME_CASCA(x1, x2, x3, y1, y2, y3, z1,
                     z2, z3, rho, t)

        ve = me.reshape(me.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VMG_CASCA = guvectorize(['float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                                    'float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                                   '(),(),(p,q),(pp,qq),(k)->(k)')(VMG_CASCA)


def MG_CASCA(rho, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vmg = GU_VMG_CASCA(rho, t, coord, connec_face, arrayfalse)
    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    return mg
