from numba import guvectorize
from numpy import shape

def KE_DKT(xg1, xg2, xg3, yg1, yg2, yg3, E, nu, t):
    '''
        PARAMETERS: xgi, ygi:   float64
                                Cartesian coordinates in the plane of the
                                surface of the DKT element.
                    E:  float64
                        Modulus of linear elasticity.
                    nu:  float64
                        Poisson Coefficient.
                    t:  float64
                        Thickness
        RETURNS:    out:    ndarray.dtype(float64)
                            Element stiffness matrix DKT in global coordinates
                            of the plane of the element surface.
    '''
    from numpy import (array, sqrt, zeros)

    l12 = sqrt((xg2 - xg1)**2. + (yg2 - yg1)**2.)

    cosxX = (xg2 - xg1) / l12
    cosxY = (yg2 - yg1) / l12

    cosyX = - cosxY
    cosyY = cosxX

    lamb = array([[1., 0., 0.], [0., cosxX, cosyX], [0., cosxY, cosyY]])

    coord_global = array([[0., 0., 0.], [xg1, xg2, xg3], [yg1, yg2, yg3]])
    coord_local = lamb.T.dot(coord_global)

    # Translating point 1 to the local coordinate O (0, 0, 0).
    dx = coord_local[1, 0]
    dy = coord_local[2, 0]
    coord_local[1, :] -= dx
    coord_local[2, :] -= dy
    coord_local[1, 0], coord_local[2, 0], coord_local[2, 1] = 0., 0., 0.

    x1, x2, x3 = coord_local[1]
    y1, y2, y3 = coord_local[2]

    x23 = x2 - x3
    y23 = y2 - y3

    x12 = x1 - x2
    y12 = y1 - y2

    x31 = x3 - x1
    y31 = y3 - y1

    A = abs(x31 * y12 - x12 * y31) * 0.5

    l2_23 = x23**2. + y23**2.
    l2_12 = x12**2. + y12**2.
    l2_31 = x31**2. + y31**2.

    p4 = - 6. * x23 / l2_23
    p5 = - 6. * x31 / l2_31
    p6 = - 6. * x12 / l2_12

    q4 = 3. * x23 * y23 / l2_23
    q5 = 3. * x31 * y31 / l2_31

    r4 = 3. * (y23**2.) / l2_23
    r5 = 3. * (y31**2.) / l2_31

    t4 = - 6. * y23 / l2_23
    t5 = - 6. * y31 / l2_31

    alfa = array([[y3 * p6, 0., -4. * y3, - y3 * p6, 0., -2. * y3, 0., 0., 0],
                  [-y3 * p6, 0., 2. * y3, y3 * p6, 0., 4. * y3, 0., 0., 0.],
                  [y3 * p5, -y3 * q5, y3 * (2. - r5), y3 * p4, y3 * q4,
                   y3 * (r4 - 2.), -y3 * (p4 + p5), y3 * (q4 - q5),
                   y3 * (r4 - r5)],
                  [-x2 * t5, x23 + x2 * r5, -x2 * q5, 0., x3, 0., x2 * t5,
                   x2 * (r5 - 1.), -x2 * q5],
                  [0., x23, 0., x2 * t4, x3 + x2 * r4, -x2 * q4, -x2 * t4,
                   x2 * (r4 - 1.), -x2 * q4],
                  [x23 * t5, x23 * (1. - r5), x23 * q5, -x3 * t4,
                   x3 * (1. - r4), x3 * q4, -x23 * t5 + x3 * t4,
                   -x23 * r5 - x3 * r4 - x2, x3 * q4 + x23 * q5],
                  [-x3 * p6 - x2 * p5, x2 * q5 + y3, -4. * x23 + x2 * r5,
                   x3 * p6, -y3, 2. * x3, x2 * p5, x2 * q5, (r5 - 2.) * x2],
                  [-x23 * p6, y3, 2. * x23, x23 * p6 + x2 * p4, -y3 + x2 * q4,
                   -4. * x3 + x2 * r4, -x2 * p4, x2 * q4, (r4 - 2.) * x2],
                  [x23 * p5 + y3 * t5, -x23 * q5 + (1. - r5) * y3,
                   (2. - r5) * x23 + y3 * q5, -x3 * p4 + y3 * t4,
                   (r4 - 1.) * y3 - x3 * q4, (2. - r4) * x3 - y3 * q4,
                   -x23 * p5 + x3 * p4 - (t4 + t5) * y3,
                   -x23 * q5 - x3 * q4 + (r4 - r5) * y3,
                   -x23 * r5 - x3 * r4 + 4. * x2 + (q5 - q4) * y3]])

    R = array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
    D = (E * t**3.) / (12. * (1. - nu**2.))

    DR = D * R

    D_ = zeros((9, 9), float)
    D_[:3, :3] = (1. / 24.) * DR
    D_[3:6, :3] = (1. / 24.) * nu * DR
    D_[:3, 3:6] = (1. / 24.) * nu * DR
    D_[3:6, 3:6] = (1. / 24.) * DR
    D_[6:, 6:] = (1. / 24.) * (1. - nu) * 0.5 * DR

    ke_local = (1. / (2. * A)) * alfa.T.dot(D_).dot(alfa)

    T = zeros((9, 9), float)
    T[:3, :3] = lamb.T
    T[3:6, 3:6] = lamb.T
    T[6:, 6:] = lamb.T

    ke_global = T.T.dot(ke_local).dot(T)

    return ke_global


def VKG_DKT(E, nu, t, coord, connec_face, arrayfalse, out):
    nglel = 9

    p, q = shape(coord)
    pp, qq = shape(connec_face)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x1, x2, x3 = coord[connec_face[i], 0]
        y1, y2, y3 = coord[connec_face[i], 1]

        ke = KE_DKT(x1, x2, x3, y1, y2, y3, E, nu, t)

        ve = ke.reshape(ke.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VKG_DKT = guvectorize(['float32, float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                          'float64, float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                         '(),(),(),(p,q),(pp,qq),(k)->(k)')(VKG_DKT)


def KG_DKT(E, nu, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vkg = GU_VKG_DKT(E, nu, t, coord, connec_face, arrayfalse)

    kg = csr_matrix((vkg, (I, J)), shape=(ngl, ngl))

    return kg
