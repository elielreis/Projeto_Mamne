def Index_Global_DKT(connec_face):
    from numpy import array

    g = 3
    nglel = 9

    no1 = g * connec_face[:, 0]
    no2 = g * connec_face[:, 1]
    no3 = g * connec_face[:, 2]

    m = array([no1, no1 + 1, no1 + 2,
               no2, no2 + 1, no2 + 2,
               no3, no3 + 1, no3 + 2])
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
