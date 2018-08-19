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


def KE_CASCA(x1, x2, x3, y1, y2, y3, z1, z2, z3, E, nu, t):
    from FUNCTIONS_EOL import LAMB
    from numpy import zeros, array


    '''
        PARAMETERS: xi, yi: float64
                            Global Cartesian coordinates in
                            3d Euclidean space.
                    E:  float64
                        Modulus of linear elasticity.
                    nu:  float64
                        Poisson Coefficient.
                    t:  float64
                        Thickness
        RETURNS:    out:    ndarray.dtype(float64)
                            Element stiffness matrix FLAT in global coordinates
                            of the plane of the element surface.
    '''

    lamb = LAMB(x1, x2, x3, y1, y2, y3, z1, z2, z3)

    T6 = zeros((6, 6), float)

    T6[:3, :3] = lamb.T
    T6[3:, 3:] = lamb.T

    T18 = zeros((18, 18), float)

    T18[:6, :6] = T6
    T18[6:12, 6:12] = T6
    T18[12:, 12:] = T6

    coord_global = array([[x1, x2, x3],
                          [y1, y2, y3],
                          [z1, z2, z3]], float)

    coord_local = lamb.T.dot(coord_global)

    x1l, x2l, x3l = coord_local[0]
    y1l, y2l, y3l = coord_local[1]

    ke_dkt = KE_DKT(x1l, x2l, x3l, y1l, y2l, y3l, E, nu, t)
    ke_cst = KE_CST(x1l, x2l, x3l, y1l, y2l, y3l, E, nu, t)

    ke_casca = zeros((18, 18), float)

    ids = [0, 1, 6, 7, 12, 13]
    for i in range(6):
        for j in range(6):
            ke_casca[ids[i], ids[j]] = ke_cst[i, j]

    ids = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    for i in range(9):
        for j in range(9):
            ke_casca[ids[i], ids[j]] = ke_dkt[i, j]

    ke_casca[5, 5] = 1.E-10
    ke_casca[11, 11] = 1.E-10
    ke_casca[17, 17] = 1.E-10

    ke_global = T18.T.dot(ke_casca).dot(T18)

    return ke_global


def VKG_CASCA(E, nu, t, coord, connec_face, arrayfalse, out):

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

        ke = KE_CASCA(x1, x2, x3, y1, y2, y3, z1,
                     z2, z3, E, nu, t)

        ve = ke.reshape(ke.size)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VKG_CASCA = guvectorize(['float32, float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                                    'float64, float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                                   '(),(),(),(p,q),(pp,qq),(k)->(k)')(VKG_CASCA)


def KG_CASCA(E, nu, t, ngl, coord, connec_face, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vkg = GU_VKG_CASCA(E, nu, t, coord, connec_face, arrayfalse)

    kg = csr_matrix((vkg, (I, J)), shape=(ngl, ngl))

    return kg

