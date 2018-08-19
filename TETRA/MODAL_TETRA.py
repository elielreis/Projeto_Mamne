from numpy import (shape, zeros, sqrt, pi)
from scipy.sparse.linalg import eigsh
from FUNCTIONS_TETRA import Index_Global_TETRA
from KG_TETRA import KG_TETRA
from MG_TETRA import MG_TETRA


class MODAL_TETRA:

    def __init__(self, coord, connec_volume, nodesr,
                 E, nu, rho, gr, k):
        self.k = k
        self.g = 3
        self.coord = coord
        self.connec_volume = connec_volume
        self.nodesr = nodesr
        self.E = E
        self.nu = nu
        self.rho = rho
        self.nelem, self.nnosel = shape(self.connec_volume)
        self.nnos = len(self.coord)
        self.ngl = self.g * self.nnos
        self.gr = gr
        self.dirr = zeros((self.nnos, self.g), int)
        for i in range(len(self.gr)):
            if gr[i]:
                self.dirr[self.nodesr, i] = 1
        self.nglel = self.g * self.nnosel

        self.I, self.J = Index_Global_TETRA(self.connec_volume)
        self.kg = KG_TETRA(
            self.E,
            self.nu,
            self.ngl,
            self.coord,
            self.connec_volume,
            self.I,
            self.J)
        self.mg = MG_TETRA(
            self.rho,
            self.ngl,
            self.coord,
            self.connec_volume,
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
            self.kgr, self.k, M=self.mgr, sigma=0, which='LM')
        self.w = sqrt(self.w2)
        self.hz = (1. / (2. * pi)) * self.w