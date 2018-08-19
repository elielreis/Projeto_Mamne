from numba import guvectorize
from numpy import shape


def KE_CST(xg1, xg2, xg3, yg1, yg2, yg3, E, nu, t):
    '''
        PARAMETERS: xgi, ygi:  float64
                                Cartesian coordinates in the plane of the
                                surface of the DKT element.
                    E:  float64
                        Modulus of linear elasticity.
                    nu:  float64
                        Poisson Coefficient.
                    t:  float64
                        Thickness
        RETURNS:    out:    ndarray.dtype(float64)
                            Element stiffness matrix CST in global coordinates
                            of the plane of the element surface.
    '''

    from numpy import array

    x23 = xg2 - xg3
    y23 = yg2 - yg3

    x12 = xg1 - xg2
    y12 = yg1 - yg2

    x31 = xg3 - xg1
    y31 = yg3 - yg1

    A = abs(x31 * y12 - x12 * y31) * 0.5

    B = (1. / (2. * A)) * array([[y23, 0., y31, 0., y12, 0.],
                                 [0., - x23, 0., - x31, 0., - x12],
                                 [- x23, y23, - x31, y31, - x12, y12]])

    D = (E / (1. - nu**2.)) * array([[1., nu, 0.],
                                    [nu, 1., 0.],
                                    [0., 0., (1. - nu) / 2.]])

    ke = (t * A) * B.T.dot(D).dot(B)

    return ke


def VKG_CST(E, nu, t, coord, connec_face, arrayfalse, out):

    nglel = 6

    p, q = shape(coord)
    pp, qq = shape(connec_face)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x1, x2, x3 = coord[connec_face[i], 0]
        y1, y2, y3 = coord[connec_face[i], 1]

        ke = KE_CST(x1, x2, x3, y1, y2, y3, E, nu, t)

        ve = ke.reshape(ke.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VKG_CST = guvectorize(['float32, float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                                   'float64, float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                                  '(),(),(),(p,q),(pp,qq),(k)->(k)')(VKG_CST)


def KG_CST(E, nu, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vkg = GU_VKG_CST(E, nu, t, coord, connec_face, arrayfalse)

    kg = csr_matrix((vkg, (I, J)), shape=(ngl, ngl))

    return kg