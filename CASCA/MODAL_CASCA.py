from numpy import (shape, zeros, sqrt, pi)
from scipy.sparse.linalg import eigsh
from FUNCTIONS_CASCA import Index_Global_CASCA
from KG_CASCA import KG_CASCA
from MG_CASCA import MG_CASCA


class MODAL_CASCA:

    def __init__(self, coord, connec_face, nodesr,
                 E, nu, t, rho, gr, k):
        self.k = k
        self.g = 6
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

        self.I, self.J = Index_Global_CASCA(self.connec_face)
        self.kg = KG_CASCA(
            self.E,
            self.nu,
            self.t,
            self.ngl,
            self.coord,
            self.connec_face,
            self.I,
            self.J)
        self.mg = MG_CASCA(
            self.rho,
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
            self.kgr, k=self.k, M=self.mgr, sigma=0, which='LM')
        self.w = sqrt(self.w2)
        self.hz = (1. / (2. * pi)) * self.w
