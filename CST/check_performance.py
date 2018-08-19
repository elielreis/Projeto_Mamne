from numpy import (shape, zeros, sqrt, pi, array, empty)
from numba import guvectorize, jit
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


class MODAL_CST:

    def __init__(self, coord, connec_face, nodesr,
                 E, nu, t, rho, gr):
        self.g = 2
        self.coord = coord
        self.connec_face = connec_face
        self.nodesr = nodesr
        self.E = E
        self.nu = nu
        self.t = t
        self.rho = rho
        self.nelem, self.nnosel = shape(self.connec_face)
        self.nnos = len(self.coord)
        self.ngl = self.g * self.nnos
        self.gr = gr
        self.dirr = zeros((self.nnos, self.g), int)
        for i in range(len(self.gr)):
            if gr[i]:
                self.dirr[self.nodesr, i] = 1
        self.nglel = self.g * self.nnosel

        self.I, self.J = Index_Global_CST(self.connec_face)
        self.kg, self.mg = KG_CST(
            self.rho,
            self.E,
            self.nu,
            self.t,
            self.ngl,
            self.coord,
            self.connec_face,
            self.I,
            self.J)
        
        self.COMPUTE()

    def COMPUTE(self):
        self.r = (self.dirr.reshape(self.ngl) - 1)**2
        self.r = self.r.astype(bool)
        self.kgr = self.kg[self.r]
        self.kgr = self.kgr[:, self.r]
        self.mgr = self.mg[self.r]
        self.mgr = self.mgr[:, self.r]

        self.w2, self.modo = eigsh(
            self.kgr, k=6, M=self.mgr, sigma=0, which='LM')
        self.w = sqrt(self.w2)
        self.hz = (1. / (2. * pi)) * self.w


def KE_CST(xg1, xg2, xg3, yg1, yg2, yg3, E, nu, t):

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


def VKG_CST(rho, E, nu, t, coord, connec_face, arrayfalse, out1, out2):

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
        me = ME_CST(x1, x2, x3, y1, y2, y3, rho, t)

        vmg = me.reshape(me.size)

        vkg = ke.reshape(ke.size)

        start = i * tt
        end = (i + 1) * tt
        out1[start:end] = vkg
        out2[start:end] = vmg


GU_VKG_CST = guvectorize(['float32, float32, float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:], float32[:]',
                          'float64, float64, float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:], float64[:]'],
                         '(),(),(),(),(p,q),(pp,qq),(k)->(k),(k)')(VKG_CST)


def KG_CST(rho, E, nu, t, ngl, coord, connec_face, I, J):

    arrayfalse = empty(I.size, int)

    vkg, vmg = GU_VKG_CST(rho, E, nu, t, coord, connec_face, arrayfalse)

    mg = csr_matrix((vmg, (I, J)), shape=(ngl, ngl))

    kg = csr_matrix((vkg, (I, J)), shape=(ngl, ngl))

    return kg, mg


def ME_CST(xg1, xg2, xg3, yg1, yg2, yg3, rho, t):

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


def Index_Global_CST(connec_face):

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

'''
def MESH_FROM_SALOME(Mesh, Cc, FileName):
    import SMESH
    from numpy import array
    import pickle
    nodes_id = list(Mesh.GetElementsByType(SMESH.NODE))
    coord = array([Mesh.GetNodeXYZ(i) for i in nodes_id], float)
    face_id = list(Mesh.GetElementsByType(SMESH.FACE))
    connec = array([Mesh.GetElemNodes(i) for i in face_id], int) - 1
    nodesr = array(Cc.GetNodeIDs(), int) - 1
    Object = {
        'coord': coord,
        'connec': connec,
        'nodesr': nodesr}
    OutFile = open(FileName, 'wb')
    pickle.dump(Object, OutFile)
    OutFile.close()
    return Object



'''

def INPUT(Dict):
    import pickle
    fileobject = open(Dict, 'r')
    model = pickle.load(fileobject)
    fileobject.close()
    return model

MODEL = INPUT('Viga2D300x50')
coord = MODEL['coord']
connec = MODEL['connec']
nodesr = MODEL['nodesr']
gr = [True, True]
E = 210.e+9
nu = 0.3
t = 0.2
rho = 2400.
import time
ini = time.time()
SOLVER = MODAL_CST(coord, connec, nodesr,
                 E, nu, t, rho, gr)
fim = time.time()
print 'DADOS'
print 'Tempo de processamento em segundos: ', fim - ini
print 'Freq. mins em Hz: ', SOLVER.hz
print 'N. de pontos nodais: ', len(coord)
print 'N. de elmentos: ', len(connec)
print 'N. grau de liberdade: ', SOLVER.ngl

#--------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


datax = [1e6,1e7,1e8]
data_NumPy = [0.0029000282287597657, 0.024699997901916505, 0.22540004253387452]
data_Numba = [0.001300048828125, 0.01100003719329834, 0.10009994506835937]

plt.plot( datax, data_NumPy, 'go', label='NumPy') # green bolinha
plt.plot( datax, data_NumPy, 'k:', color='orange') # linha pontilha orange

plt.plot( datax, data_Numba, 'r^', label='NumPy + Numba') # red triangulo
plt.plot( datax, data_Numba, 'k--', color='blue')  # linha tracejada azul

#plt.axis([-10, 60, 0, 11])

plt.grid(True)
plt.xlabel("N. de elementos da matriz")
plt.ylabel("Tempo de exec. em segundos")
plt.legend()
plt.show()