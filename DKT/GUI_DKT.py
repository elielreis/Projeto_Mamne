def GUI_DKT(context):
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
        QLineEdit)
    from numpy import array, zeros
    from MODAL_DKT import MODAL_DKT
    import pandas as pd

    study = context.study
    studyId = context.studyId
    sg = context.sg
    smesh = smeshBuilder.New(salome.myStudy)

    class APP_MODAL_DKT(QDialog):
        def __init__(self, parent=None):
            super(APP_MODAL_DKT, self).__init__()

            self.init_ui()
            self.show()

            self.mesh_b.clicked.connect(self.mesh_b_click)
            self.nodesCc_b.clicked.connect(self.nodesCc_b_click)
            self.compute.clicked.connect(
                lambda: self.compute_click(
                    self.z.isChecked(),
                    self.rx.isChecked(),
                    self.ry.isChecked(),
                    self.E_l.text(),
                    self.nu_l.text(),
                    self.rho_l.text(),
                    self.t_l.text()))

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
            self.z = QCheckBox('Uz')
            self.rx = QCheckBox('Rx')
            self.ry = QCheckBox('Ry')

            h_boxNodesCc = QHBoxLayout()
            h_boxNodesCc.addWidget(self.nodesCc_lb)
            h_boxNodesCc.addWidget(self.nodesCc_l)
            h_boxNodesCc.addWidget(self.nodesCc_b)

            h_boxCheck = QHBoxLayout()
            h_boxCheck.addWidget(self.z)
            h_boxCheck.addWidget(self.rx)
            h_boxCheck.addWidget(self.ry)

            v_boxNodesCc = QVBoxLayout()
            v_boxNodesCc.addLayout(h_boxNodesCc)
            v_boxNodesCc.addLayout(h_boxCheck)

            self.div2 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.E_lb = QLabel('MODULUS OF ELASTICITY')
            self.E_l = QLineEdit()
            self.nu_lb = QLabel('COEF. POISSON')
            self.nu_l = QLineEdit()
            self.rho_lb = QLabel('DENSITY OF MASSES')
            self.rho_l = QLineEdit()
            self.t_lb = QLabel('THICKNESS')
            self.t_l = QLineEdit()

            h_box_E = QHBoxLayout()
            h_box_E.addWidget(self.E_lb)
            h_box_E.addWidget(self.E_l)

            h_box_nu = QHBoxLayout()
            h_box_nu.addWidget(self.nu_lb)
            h_box_nu.addWidget(self.nu_l)

            h_box_rho = QHBoxLayout()
            h_box_rho.addWidget(self.rho_lb)
            h_box_rho.addWidget(self.rho_l)

            h_box_t = QHBoxLayout()
            h_box_t.addWidget(self.t_lb)
            h_box_t.addWidget(self.t_l)

            self.div3 = QLabel(
                '---------------------------------------------------------------------------------------------------------')

            self.compute = QPushButton('COMPUTE')

            v_box = QVBoxLayout()
            v_box.addLayout(h_boxMesh)
            v_box.addWidget(self.div)
            v_box.addLayout(v_boxNodesCc)
            v_box.addWidget(self.div2)
            v_box.addLayout(h_box_E)
            v_box.addLayout(h_box_nu)
            v_box.addLayout(h_box_rho)
            v_box.addLayout(h_box_t)
            v_box.addWidget(self.div3)
            v_box.addWidget(self.compute)

            self.setLayout(v_box)
            self.setWindowTitle('CODE_MAMNE/MODAL_DKT')

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
                    face_id = list(self.mesh.GetElementsByType(SMESH.FACE))
                    self.connec_face = array(
                        [self.mesh.GetElemNodes(i) for i in face_id], int) - 1

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

        def compute_click(self, Uz, Rx, Ry, E, nu, rho, t):

            self.gr = [Uz, Rx, Ry]
            self.E = float(E)
            self.nu = float(nu)
            self.rho = float(rho)
            self.t = float(t)

            self.MODOS = MODAL_DKT(self.coord, self.connec_face, self.nodesr,
                                   self.E, self.nu, self.t, self.rho, self.gr)

            raw_data = {'NATURAL FREQUENCIES': self.MODOS.hz}
            df = pd.DataFrame(raw_data, columns=['NATURAL FREQUENCIES'])
            df.to_csv('MODAL_DKT.csv')

    app = APP_MODAL_DKT()
    app.exec_()
