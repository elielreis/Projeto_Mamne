from numba import guvectorize
from numpy import shape, array
from numpy.linalg import det


def VME_TETRA_E(rho, volume):
    from numpy import array, zeros

    vme = zeros(24**2, float)

    IndexVmeInVmeE = array([ 
            0,   1,   2,   6,   7,   8,  12,  13,  14,  18,  19,  20,  24,
            25,  26,  30,  31,  32,  36,  37,  38,  42,  43,  44,  48,  49,
            50,  54,  55,  56,  60,  61,  62,  66,  67,  68, 144, 145, 146,
            150, 151, 152, 156, 157, 158, 162, 163, 164, 168, 169, 170, 174,
            175, 176, 180, 181, 182, 186, 187, 188, 192, 193, 194, 198, 199,
            200, 204, 205, 206, 210, 211, 212, 288, 289, 290, 294, 295, 296,
            300, 301, 302, 306, 307, 308, 312, 313, 314, 318, 319, 320, 324,
            325, 326, 330, 331, 332, 336, 337, 338, 342, 343, 344, 348, 349,
            350, 354, 355, 356, 432, 433, 434, 438, 439, 440, 444, 445, 446,
            450, 451, 452, 456, 457, 458, 462, 463, 464, 468, 469, 470, 474,
            475, 476, 480, 481, 482, 486, 487, 488, 492, 493, 494, 498, 499,
            500])
    vme[IndexVmeInVmeE] = ((rho * volume) / 20.) * \
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


def VMG_TETRA_E(rho, coord, connec_volume, arrayfalse, out):

    nglel = 24

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

        ve = VME_TETRA_E(rho, volume)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VMG_TETRA_E = guvectorize(['float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                            'float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                           '(),(p,q),(pp,qq),(k)->(k)')(VMG_TETRA_E)


def MG_TETRA_E(rho, ngl, coord, connec_volume, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vmg = GU_VMG_TETRA_E(rho, coord, connec_volume, arrayfalse)
    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    return mg
