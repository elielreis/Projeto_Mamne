from numpy import (shape, zeros, sqrt, pi)
from scipy.sparse.linalg import eigsh
from FUNCTIONS_DKT import Index_Global_DKT
from KG_DKT import KG_DKT
from MG_DKT import MG_DKT


class MODAL_DKT:

    def __init__(self, coord, connec_face, nodesr,
                 E, nu, t, rho, gr):
        self.g = 3
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

        self.I, self.J = Index_Global_DKT(self.connec_face)
        self.kg = KG_DKT(
            self.E,
            self.nu,
            self.t,
            self.ngl,
            self.coord,
            self.connec_face,
            self.I,
            self.J)
        self.mg = MG_DKT(
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
            self.kgr, k=6, M=self.mgr, sigma=0, which='LM')
        self.w = sqrt(self.w2)
        self.hz = (1. / (2. * pi)) * self.w
