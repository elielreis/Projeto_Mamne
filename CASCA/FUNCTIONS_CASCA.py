def Index_Global_CASCA(connec_face):
    from numpy import array

    g = 6
    nglel = 18

    no1 = g * connec_face[:, 0]
    no2 = g * connec_face[:, 1]
    no3 = g * connec_face[:, 2]

    m = array([no1, no1 + 1, no1 + 2, no1 + 3, no1 + 4, no1 + 5,
               no2, no2 + 1, no2 + 2, no2 + 3, no2 + 4, no2 + 5,
               no3, no3 + 1, no3 + 2, no3 + 3, no3 + 4, no3 + 5])
    m = m.T

    j = array(list(range(nglel)) * nglel).reshape((nglel, nglel))
    i = j.T
    j = j.reshape(j.size)
    i = i.reshape(i.size)

    J = m[:, j]
    J = J.reshape(J.size)
    I = m[:, i]
    I = I.reshape(I.size)

    return I, J


def LAMB(x1, x2, x3, y1, y2, y3, z1, z2, z3):
    from numpy import array, cross
    from numpy.linalg import norm, inv


    '''
        PARAMETERS: xi, yi, zi: float64
                                Cartesian coordinates global.
        RETURN: out:    ndarray.dtype(float64)
                        Matrix transformation of base between global and local.
    '''

    Vx = array([x2 - x1,
                y2 - y1,
                z2 - z1])
    lx = norm(Vx)
    X = Vx / lx

    Vr = array([x3 - x1,
                y3 - y1,
                z3 - z1])
    lr = norm(Vr)
    R = Vr / lr

    Vz = cross(X, R)
    lz = norm(Vz)
    Z = Vz / lz

    Vy = cross(Z, X)
    ly = norm(Vy)
    Y = Vy / ly

    V = array([X, Y, Z])

    lamb = inv(V)

    return lamb