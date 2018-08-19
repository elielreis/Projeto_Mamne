from numpy import (shape, zeros, sqrt, pi)
from scipy.sparse.linalg import eigsh
from FUNCTIONS_EOL import Index_Global_CASCA, Index_Global_TETRA_E
from KG_TETRA_E import KG_TETRA_E
from MG_TETRA_E import MG_TETRA_E
from KG_CASCA import KG_CASCA
from MG_CASCA import MG_CASCA


class MODAL_EOL:

    def __init__(self, coord, nodesr, gr,
                    connec_nacele, E_nacele, nu_nacele, rho_nacele,
                    connec_hub, E_hub, nu_hub, rho_hub, t_hub,
                    connec_blades, E_blades, nu_blades, rho_blades, t_blades,
                    connec_tower, E_tower, nu_tower, rho_tower, t_tower, k):
        self.k = int(k)
        self.g = 6
        self.coord = coord
        self.nodesr = nodesr

        self.connec_nacele, self.E_nacele, self.nu_nacele, self.rho_nacele = connec_nacele, E_nacele, nu_nacele, rho_nacele
        self.connec_hub, self.E_hub, self.nu_hub, self.rho_hub, self.t_hub = connec_hub, E_hub, nu_hub, rho_hub, t_hub
        self.connec_blades, self.E_blades, self.nu_blades, self.rho_blades, self.t_blades = connec_blades, E_blades, nu_blades, rho_blades, t_blades
        self.connec_tower, self.E_tower, self.nu_tower, self.rho_tower, self.t_tower = connec_tower, E_tower, nu_tower, rho_tower, t_tower
       
        self.nnos = len(self.coord)
        self.ngl = self.g * self.nnos
        self.gr = gr
        self.dirr = zeros((self.nnos, self.g), int)
        for i in range(len(self.gr)):
            if gr[i]:
                self.dirr[self.nodesr, i] = 1
        
        self.I_nacele, self.J_nacele = Index_Global_TETRA_E(self.connec_nacele)
        self.I_hub, self.J_hub = Index_Global_CASCA(self.connec_hub)
        self.I_blades, self.J_blades = Index_Global_CASCA(self.connec_blades)
        self.I_tower, self.J_tower = Index_Global_CASCA(self.connec_tower)

        self.kg_nacele = KG_TETRA_E(
            self.E_nacele,
            self.nu_nacele,
            self.ngl,
            self.coord,
            self.connec_nacele,
            self.I_nacele,
            self.J_nacele)
        self.mg_nacele = MG_TETRA_E(
            self.rho_nacele,
            self.ngl,
            self.coord,
            self.connec_nacele,
            self.I_nacele,
            self.J_nacele)

        self.kg_hub = KG_CASCA(
            self.E_hub,
            self.nu_hub,
            self.t_hub,
            self.ngl,
            self.coord,
            self.connec_hub,
            self.I_hub,
            self.J_hub)
        self.mg_hub = MG_CASCA(
            self.rho_hub,
            self.t_hub,
            self.ngl,
            self.coord,
            self.connec_hub,
            self.I_hub,
            self.J_hub)

        self.kg_blades = KG_CASCA(
            self.E_blades,
            self.nu_blades,
            self.t_blades,
            self.ngl,
            self.coord,
            self.connec_blades,
            self.I_blades,
            self.J_blades)
        self.mg_blades = MG_CASCA(
            self.rho_blades,
            self.t_blades,
            self.ngl,
            self.coord,
            self.connec_blades,
            self.I_blades,
            self.J_blades)

        self.kg_tower = KG_CASCA(
            self.E_tower,
            self.nu_tower,
            self.t_tower,
            self.ngl,
            self.coord,
            self.connec_tower,
            self.I_tower,
            self.J_tower)
        self.mg_tower = MG_CASCA(
            self.rho_tower,
            self.t_tower,
            self.ngl,
            self.coord,
            self.connec_tower,
            self.I_tower,
            self.J_tower)

        self.kg = self.kg_nacele + self.kg_hub + self.kg_blades + self.kg_tower
        self.mg = self.mg_nacele + self.mg_hub + self.mg_blades + self.mg_tower

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