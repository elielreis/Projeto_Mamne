def GUI_EOL(context):
    import salome
    from salome.smesh import smeshBuilder
    import SMESH
    from PyQt5.QtWidgets import QDialog
    from PyQt5.QtWidgets import (
        QPushButton,
        QLabel,
        QCheckBox,
        QHBoxLayout,
        QVBoxLayout,
        QLineEdit,
        QFileDialog)
    from numpy import array, zeros
    import pyvtk
    from MODAL_EOL import MODAL_EOL
    import pandas as pd

    study = context.study
    studyId = context.studyId
    sg = context.sg
    smesh = smeshBuilder.New(salome.myStudy)

    class APP_MODAL_EOL(QDialog):
        def __init__(self, parent=None):
            super(APP_MODAL_EOL, self).__init__()

            self.init_ui()
            self.show()

            self.mesh_b.clicked.connect(self.mesh_b_click)
            self.nodesCc_b.clicked.connect(self.nodesCc_b_click)
            self.nacele_b.clicked.connect(self.nacele_b_click)
            self.hub_b.clicked.connect(self.hub_b_click)
            self.blades_b.clicked.connect(self.blades_b_click)
            self.tower_b.clicked.connect(self.tower_b_click)
            
            self.compute.clicked.connect(
                lambda: self.compute_click(
                    self.x.isChecked(),
                    self.y.isChecked(),
                    self.z.isChecked(),
                    self.rx.isChecked(),
                    self.ry.isChecked(),
                    self.rz.isChecked(),
                    self.nacele_E_l.text(),
                    self.nacele_nu_l.text(),
                    self.nacele_rho_l.text(),
                    self.hub_E_l.text(),
                    self.hub_nu_l.text(),
                    self.hub_rho_l.text(),
                    self.hub_t_l.text(),
                    self.blades_E_l.text(),
                    self.blades_nu_l.text(),
                    self.blades_rho_l.text(),
                    self.blades_t_l.text(),
                    self.tower_E_l.text(),
                    self.tower_nu_l.text(),
                    self.tower_rho_l.text(),
                    self.tower_t_l.text(),
                    self.k_l.text()))

            self.save.clicked.connect(lambda: self.save_vtk(
                        self.modei_l.text(),
                        self.scale_l.text()))

        def init_ui(self):
            self.mesh_b = QPushButton('MESH')
            self.mesh_l = QLineEdit()

            h_boxMesh = QHBoxLayout()
            h_boxMesh.addWidget(self.mesh_b)
            h_boxMesh.addWidget(self.mesh_l)

            self.div = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.nodesCc_lb = QLabel('BOUNDARY CONDITIONS')
            self.nodesCc_l = QLineEdit()
            self.nodesCc_b = QPushButton('GROUP OF NODES')
            self.x = QCheckBox('Ux')
            self.y = QCheckBox('Uy')
            self.z = QCheckBox('Uz')
            self.rx = QCheckBox('Rx')
            self.ry = QCheckBox('Ry')
            self.rz = QCheckBox('Rz')

            h_boxNodesCc = QHBoxLayout()
            h_boxNodesCc.addWidget(self.nodesCc_lb)
            h_boxNodesCc.addWidget(self.nodesCc_l)
            h_boxNodesCc.addWidget(self.nodesCc_b)

            h_boxCheck = QHBoxLayout()
            h_boxCheck.addWidget(self.x)
            h_boxCheck.addWidget(self.y)
            h_boxCheck.addWidget(self.z)
            h_boxCheck.addWidget(self.rx)
            h_boxCheck.addWidget(self.ry)
            h_boxCheck.addWidget(self.rz)

            v_boxNodesCc = QVBoxLayout()
            v_boxNodesCc.addLayout(h_boxNodesCc)
            v_boxNodesCc.addLayout(h_boxCheck)

            self.div2 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.hub_lb = QLabel('HUB')
            self.hub_b = QPushButton('GROUP OF ELEMENTS')
            self.hub_l = QLineEdit()
            self.hub_E_lb = QLabel('MODULUS OF ELASTICITY')
            self.hub_E_l = QLineEdit()
            self.hub_nu_lb = QLabel('COEF. POISSON')
            self.hub_nu_l = QLineEdit()
            self.hub_rho_lb = QLabel('DENSITY OF MASSES')
            self.hub_rho_l = QLineEdit()
            self.hub_t_lb = QLabel('THICKNESS')
            self.hub_t_l = QLineEdit()

            h_box_hub = QHBoxLayout()
            h_box_hub.addWidget(self.hub_lb)
            h_box_hub.addWidget(self.hub_l)
            h_box_hub.addWidget(self.hub_b)

            h_box_hub_E = QHBoxLayout()
            h_box_hub_E.addWidget(self.hub_E_lb)
            h_box_hub_E.addWidget(self.hub_E_l)

            h_box_hub_nu = QHBoxLayout()
            h_box_hub_nu.addWidget(self.hub_nu_lb)
            h_box_hub_nu.addWidget(self.hub_nu_l)

            h_box_hub_rho = QHBoxLayout()
            h_box_hub_rho.addWidget(self.hub_rho_lb)
            h_box_hub_rho.addWidget(self.hub_rho_l)

            h_box_hub_t = QHBoxLayout()
            h_box_hub_t.addWidget(self.hub_t_lb)
            h_box_hub_t.addWidget(self.hub_t_l)

            v_box_hub = QVBoxLayout()
            v_box_hub.addLayout(h_box_hub)
            v_box_hub.addLayout(h_box_hub_E)
            v_box_hub.addLayout(h_box_hub_nu)
            v_box_hub.addLayout(h_box_hub_rho)
            v_box_hub.addLayout(h_box_hub_t)

            self.div3 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.blades_lb = QLabel('BLADES')
            self.blades_b = QPushButton('GROUP OF ELEMENTS')
            self.blades_l = QLineEdit()
            self.blades_E_lb = QLabel('MODULUS OF ELASTICITY')
            self.blades_E_l = QLineEdit()
            self.blades_nu_lb = QLabel('COEF. POISSON')
            self.blades_nu_l = QLineEdit()
            self.blades_rho_lb = QLabel('DENSITY OF MASSES')
            self.blades_rho_l = QLineEdit()
            self.blades_t_lb = QLabel('THICKNESS')
            self.blades_t_l = QLineEdit()

            h_box_blades = QHBoxLayout()
            h_box_blades.addWidget(self.blades_lb)
            h_box_blades.addWidget(self.blades_l)
            h_box_blades.addWidget(self.blades_b)

            h_box_blades_E = QHBoxLayout()
            h_box_blades_E.addWidget(self.blades_E_lb)
            h_box_blades_E.addWidget(self.blades_E_l)

            h_box_blades_nu = QHBoxLayout()
            h_box_blades_nu.addWidget(self.blades_nu_lb)
            h_box_blades_nu.addWidget(self.blades_nu_l)

            h_box_blades_rho = QHBoxLayout()
            h_box_blades_rho.addWidget(self.blades_rho_lb)
            h_box_blades_rho.addWidget(self.blades_rho_l)

            h_box_blades_t = QHBoxLayout()
            h_box_blades_t.addWidget(self.blades_t_lb)
            h_box_blades_t.addWidget(self.blades_t_l)

            v_box_blades = QVBoxLayout()
            v_box_blades.addLayout(h_box_blades)
            v_box_blades.addLayout(h_box_blades_E)
            v_box_blades.addLayout(h_box_blades_nu)
            v_box_blades.addLayout(h_box_blades_rho)
            v_box_blades.addLayout(h_box_blades_t)

            self.div4 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.tower_lb = QLabel('TOWER')
            self.tower_b = QPushButton('GROUP OF ELEMENTS')
            self.tower_l = QLineEdit()
            self.tower_E_lb = QLabel('MODULUS OF ELASTICITY')
            self.tower_E_l = QLineEdit()
            self.tower_nu_lb = QLabel('COEF. POISSON')
            self.tower_nu_l = QLineEdit()
            self.tower_rho_lb = QLabel('DENSITY OF MASSES')
            self.tower_rho_l = QLineEdit()
            self.tower_t_lb = QLabel('THICKNESS')
            self.tower_t_l = QLineEdit()

            h_box_tower = QHBoxLayout()
            h_box_tower.addWidget(self.tower_lb)
            h_box_tower.addWidget(self.tower_l)
            h_box_tower.addWidget(self.tower_b)

            h_box_tower_E = QHBoxLayout()
            h_box_tower_E.addWidget(self.tower_E_lb)
            h_box_tower_E.addWidget(self.tower_E_l)

            h_box_tower_nu = QHBoxLayout()
            h_box_tower_nu.addWidget(self.tower_nu_lb)
            h_box_tower_nu.addWidget(self.tower_nu_l)

            h_box_tower_rho = QHBoxLayout()
            h_box_tower_rho.addWidget(self.tower_rho_lb)
            h_box_tower_rho.addWidget(self.tower_rho_l)

            h_box_tower_t = QHBoxLayout()
            h_box_tower_t.addWidget(self.tower_t_lb)
            h_box_tower_t.addWidget(self.tower_t_l)

            v_box_tower = QVBoxLayout()
            v_box_tower.addLayout(h_box_tower)
            v_box_tower.addLayout(h_box_tower_E)
            v_box_tower.addLayout(h_box_tower_nu)
            v_box_tower.addLayout(h_box_tower_rho)
            v_box_tower.addLayout(h_box_tower_t)

            self.div5 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.nacele_lb = QLabel('NACELE')
            self.nacele_b = QPushButton('GROUP OF ELEMENTS')
            self.nacele_l = QLineEdit()
            self.nacele_E_lb = QLabel('MODULUS OF ELASTICITY')
            self.nacele_E_l = QLineEdit()
            self.nacele_nu_lb = QLabel('COEF. POISSON')
            self.nacele_nu_l = QLineEdit()
            self.nacele_rho_lb = QLabel('DENSITY OF MASSES')
            self.nacele_rho_l = QLineEdit()

            h_box_nacele = QHBoxLayout()
            h_box_nacele.addWidget(self.nacele_lb)
            h_box_nacele.addWidget(self.nacele_l)
            h_box_nacele.addWidget(self.nacele_b)

            h_box_nacele_E = QHBoxLayout()
            h_box_nacele_E.addWidget(self.nacele_E_lb)
            h_box_nacele_E.addWidget(self.nacele_E_l)

            h_box_nacele_nu = QHBoxLayout()
            h_box_nacele_nu.addWidget(self.nacele_nu_lb)
            h_box_nacele_nu.addWidget(self.nacele_nu_l)

            h_box_nacele_rho = QHBoxLayout()
            h_box_nacele_rho.addWidget(self.nacele_rho_lb)
            h_box_nacele_rho.addWidget(self.nacele_rho_l)

            v_box_nacele = QVBoxLayout()
            v_box_nacele.addLayout(h_box_nacele)
            v_box_nacele.addLayout(h_box_nacele_E)
            v_box_nacele.addLayout(h_box_nacele_nu)
            v_box_nacele.addLayout(h_box_nacele_rho)

            self.div6 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.k_lb = QLabel('MAX MODOS TO FIND')
            self.k_l = QLineEdit()

            h_box_k = QHBoxLayout()
            h_box_k.addWidget(self.k_lb)
            h_box_k.addWidget(self.k_l)

            self.compute = QPushButton('COMPUTE')

            self.modei_lb = QLabel('MODE')
            self.modei_l = QLineEdit()
            self.scale_lb = QLabel('SCALE')
            self.scale_l = QLineEdit()

            h_box_vtk = QHBoxLayout()
            h_box_vtk.addWidget(self.modei_lb)
            h_box_vtk.addWidget(self.modei_l)
            h_box_vtk.addWidget(self.scale_lb)
            h_box_vtk.addWidget(self.scale_l)
            
            self.save = QPushButton('SAVE')

            v_box = QVBoxLayout()
            v_box.addLayout(h_boxMesh)
            v_box.addWidget(self.div)
            v_box.addLayout(v_boxNodesCc)
            v_box.addWidget(self.div2)
            v_box.addLayout(v_box_nacele)
            v_box.addWidget(self.div3)
            v_box.addLayout(v_box_hub)
            v_box.addWidget(self.div4)
            v_box.addLayout(v_box_blades)
            v_box.addWidget(self.div5)
            v_box.addLayout(v_box_tower)
            v_box.addWidget(self.div6)
            v_box.addLayout(h_box_k)
            v_box.addWidget(self.compute)
            v_box.addLayout(h_box_vtk)
            v_box.addWidget(self.save)

            self.setLayout(v_box)
            self.setWindowTitle('CODE_MAMNE/MODAL_EOL')

        def mesh_b_click(self):
            # sg.getObjectBrowser().selectionChanged.connect(self.select)
            self.mesh = None
            objId = salome.sg.getSelected(0)
            if objId:
                mm = study.FindObjectID(objId).GetObject()
                mesh = None
                try:
                    mm.Load()
                    mesh = mm
                except BaseException:
                    mesh = None
                    self.mesh_l.setText('Select a valid mesh')
                    pass
                if mesh:
                    name = smeshBuilder.GetName(mm)
                    self.mesh_l.setText(name)
                    self.mesh = mm
                    nodes_id = list(self.mesh.GetElementsByType(SMESH.NODE))
                    self.coord = array([self.mesh.GetNodeXYZ(i)
                                        for i in nodes_id], float)

        def nodesCc_b_click(self):
            # sg.getObjectBrowser().selectionChanged.connect(self.select)
            self.cc = None
            objId = salome.sg.getSelected(0)
            if objId:
                cc = study.FindObjectID(objId).GetObject()
                Cc = None
                try:
                    cc.GetNodeIDs()
                    Cc = cc
                except BaseException:
                    Cc = None
                    self.nodesCc_l.setText('Select a valid group')
                    pass
                if Cc:
                    name = cc.GetName()
                    self.nodesCc_l.setText(name)
                    self.nodesr = array(cc.GetNodeIDs(), int) - 1

        def hub_b_click(self):
            objId = salome.sg.getSelected(0)
            if objId:
                group = study.FindObjectID(objId).GetObject()
                Group = None
                try:
                    ids = list(group.GetIDs())
                    A = zeros((5, 3), int)
                    A[2] = self.mesh.GetElemNodes(ids[0])
                    Group = ids
                except BaseException:
                    Group = None
                    self.hub_l.setText('Select a valid group')
                    pass
                if Group:
                    name = group.GetName()
                    self.hub_l.setText(name)
                    self.connec_hub = array(
                        [self.mesh.GetElemNodes(i) for i in Group], int) - 1

        def blades_b_click(self):
            objId = salome.sg.getSelected(0)
            if objId:
                group = study.FindObjectID(objId).GetObject()
                Group = None
                try:
                    ids = list(group.GetIDs())
                    A = zeros((5, 3), int)
                    A[2] = self.mesh.GetElemNodes(ids[0])
                    Group = ids
                except BaseException:
                    Group = None
                    self.blades_l.setText('Select a valid group')
                    pass
                if Group:
                    name = group.GetName()
                    self.blades_l.setText(name)
                    self.connec_blades = array(
                        [self.mesh.GetElemNodes(i) for i in Group], int) - 1

        def tower_b_click(self):
            objId = salome.sg.getSelected(0)
            if objId:
                group = study.FindObjectID(objId).GetObject()
                Group = None
                try:
                    ids = list(group.GetIDs())
                    A = zeros((5, 3), int)
                    A[2] = self.mesh.GetElemNodes(ids[0])
                    Group = ids
                except BaseException:
                    Group = None
                    self.tower_l.setText('Select a valid group')
                    pass
                if Group:
                    name = group.GetName()
                    self.tower_l.setText(name)
                    self.connec_tower = array(
                        [self.mesh.GetElemNodes(i) for i in Group], int) - 1

        def nacele_b_click(self):
            objId = salome.sg.getSelected(0)
            if objId:
                group = study.FindObjectID(objId).GetObject()
                Group = None
                try:
                    ids = list(group.GetIDs())
                    A = zeros((5, 4), int)
                    A[2] = self.mesh.GetElemNodes(ids[0])
                    Group = ids
                except BaseException:
                    Group = None
                    self.nacele_l.setText('Select a valid group')
                    pass
                if Group:
                    name = group.GetName()
                    self.nacele_l.setText(name)
                    self.connec_nacele = array(
                        [self.mesh.GetElemNodes(i) for i in Group], int) - 1

        def compute_click(self, Ux, Uy, Uz, Rx, Ry, Rz,
                          E_nacele, nu_nacele, rho_nacele,
                          E_hub, nu_hub, rho_hub, t_hub,
                          E_blades, nu_blades, rho_blades, t_blades,
                          E_tower, nu_tower, rho_tower, t_tower, k):

            self.compute.setText('COMPUTING')

            self.gr = [Ux, Uy, Uz, Rx, Ry, Rz]
            self.E_nacele = float(E_nacele)
            self.nu_nacele = float(nu_nacele)
            self.rho_nacele = float(rho_nacele)
            self.E_hub = float(E_hub)
            self.nu_hub = float(nu_hub)
            self.rho_hub = float(rho_hub)
            self.t_hub = float(t_hub)
            self.E_blades = float(E_blades)
            self.nu_blades = float(nu_blades)
            self.rho_blades = float(rho_blades)
            self.t_blades = float(t_blades)
            self.E_tower = float(E_tower)
            self.nu_tower = float(nu_tower)
            self.rho_tower = float(rho_tower)
            self.t_tower = float(t_tower)
            self.k = int(k)

            self.MODOS = MODAL_EOL(
                self.coord,
                self.nodesr,
                self.gr,
                self.connec_nacele,
                self.E_nacele,
                self.nu_nacele,
                self.rho_nacele,
                self.connec_hub,
                self.E_hub,
                self.nu_hub,
                self.rho_hub,
                self.t_hub,
                self.connec_blades,
                self.E_blades,
                self.nu_blades,
                self.rho_blades,
                self.t_blades,
                self.connec_tower,
                self.E_tower,
                self.nu_tower,
                self.rho_tower,
                self.t_tower,
                self.k)

            path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))+'/'
            raw_data = {'NATURAL FREQUENCIES': self.MODOS.hz}
            df = pd.DataFrame(raw_data, columns=['NATURAL FREQUENCIES'])
            df.to_csv(path+'MODAL_EOL.csv')

            self.compute.setText('COMPUTE')

        def save_vtk(self, modei, scale):
            path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))+'/'
            print path
            
            U = self.MODOS.coord.copy()
            nnos = self.MODOS.nnos
            ngl = self.MODOS.ngl
            nodesr = self.MODOS.nodesr
            nodesl = list(set(range(nnos))-set(nodesr))
            scale = float(scale)
            modei = int(modei)-1
            g = self.MODOS.g
            U_i = zeros((nnos,g), float)
            U_i[nodesl] = self.MODOS.modo[:,modei].reshape(self.MODOS.modo[:,modei].size/g,g)
            Ux, Uy, Uz = U_i[:,0].copy(), U_i[:,1].copy(), U_i[:,2].copy()
            Rx, Ry, Rz = U_i[:,3].copy(), U_i[:,4].copy(), U_i[:,5].copy()
            U_i = U_i*scale
            U=U+U_i[:,[0, 1, 2]]

            connec_volume = self.MODOS.connec_nacele
            connec_face = array(self.MODOS.connec_hub.tolist() + self.MODOS.connec_tower.tolist() + self.MODOS.connec_blades.tolist())

            vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid (U, triangle=connec_face, tetra=connec_volume),
                      pyvtk.PointData(pyvtk.Scalars(Ux,name='Ux'),
                                      pyvtk.Scalars(Uy,name='Uy'),
                                      pyvtk.Scalars(Uz,name='Uz'),
                                      pyvtk.Scalars(Rx,name='Rx'),
                                      pyvtk.Scalars(Ry,name='Ry'),
                                      pyvtk.Scalars(Rz,name='Rz')))

            vtk.tofile(path+'MODAL_'+str(modei+1))

    app = APP_MODAL_EOL()
    app.exec_()
