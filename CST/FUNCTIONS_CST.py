def Index_Global_CST(connec_face):
    from numpy import array

    g = 2
    nglel = 6

    no1 = g * connec_face[:, 0]
    no2 = g * connec_face[:, 1]
    no3 = g * connec_face[:, 2]

    m = array([no1, no1 + 1,
               no2, no2 + 1,
               no3, no3 + 1])
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




