# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QT_design\QT_design.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QTextEdit
from pyqtgraph import GraphicsLayoutWidget
import pyqtgraph.opengl as gl

class btn_class(QtWidgets.QPushButton):
    def __init__(self, widget, obj_name, loc_x, loc_y, w, h,group):
        super(btn_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w , h )
        self.setFont(font)
        self.setObjectName(obj_name)
        group.addButton(self)
class radio_class(QtWidgets.QRadioButton):
    def __init__(self,widget, obj_name, loc_x, loc_y, w, h,group,id):
        self,super(radio_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w, h)
        self.setFont(font)
        self.setObjectName(obj_name)
        group.addButton(self,id)

class label_class(QtWidgets.QLabel):
    def __init__(self, widget, obj_name, loc_x, loc_y, w, h):
        super(label_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w, h)
        self.setFont(font)
        self.setObjectName(obj_name)

class textEdit_class(QTextEdit):
    def __init__(self, widget, obj_name, loc_x, loc_y, w, h):
        super(textEdit_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w, h)
        self.setFont(font)
        self.setObjectName(obj_name)

class Ui_MainWindow(object):
    def __init__(self):
        self.btn_group = QtGui.QButtonGroup()
        self.radio_btn_group = QtGui.QButtonGroup()


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1850, 1000)
        font = QtGui.QFont()
        font.setFamily("MS Gothic")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        path1 =470;path2=510;path3=550
        #----------widgets setup -------------------
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 60, 250, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = GraphicsLayoutWidget(self.centralwidget)
        # self.graphicsView_2 = gl.GLViewWidget(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(300, 60, 400, 400))
        self.graphicsView_2.setObjectName("graphicsView_2")
        # self.graphicsView_3 = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView_3 = gl.GLViewWidget(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(720, 60, 400, 400))
        self.graphicsView_3.setObjectName("graphicsView_3")
        # ----------label setup-----------
        self.label = label_class(self.centralwidget,"label",80, 20, 200, 30)
        self.label_2 = label_class(self.centralwidget,"label_2",430, 20, 200, 30)
        self.label_pd = label_class(self.centralwidget,"label_pd",880, 20, 200, 30)
        self.label_file = label_class(self.centralwidget,"label_file",30, path3, 150, 30)
        self.label_cam1 = label_class(self.centralwidget,"label_cam1",30, path1, 150, 30)
        self.label_cam2 = label_class(self.centralwidget,"label_cam2",30, path2, 150, 30)
        #----------btn setup-----------
        self.pd_btn = btn_class(self.centralwidget,"pd_btn",530, 670, 90, 30,self.btn_group)
        self.browse_btn  = btn_class(self.centralwidget,"browse_btn",560,path3,90,30,self.btn_group)
        self.browse_cam1_btn  = btn_class(self.centralwidget,"browse_btn",560,path1,90,30,self.btn_group)
        self.browse_cam2_btn  = btn_class(self.centralwidget,"browse_btn",560,path2,90,30,self.btn_group)
        self.load_btn = btn_class(self.centralwidget, "browse_btn", 30, 670, 90, 30,self.btn_group)
        self.start_btn  = btn_class(self.centralwidget,"start_btn",130,670,90,30,self.btn_group)
        self.stop_btn  = btn_class(self.centralwidget,"stop_btn",230,670,90,30,self.btn_group)
        self.next_btn  = btn_class(self.centralwidget,"next_btn",330,670,90,30,self.btn_group)
        self.pre_btn  = btn_class(self.centralwidget,"pre_btn",430,670,90,30,self.btn_group)
        # ----------rai/Beamforming rai / Cfar+static_rm-----------
        self.orgin_rai = radio_class(self.centralwidget,"orgin_rai",670,600,200,30,self.radio_btn_group,0)
        self.beam_rai = radio_class(self.centralwidget,"beam_rai",670,640,200,30,self.radio_btn_group,1)
        self.Cfar_rai = radio_class(self.centralwidget,"static_rai",670,680,200,30,self.radio_btn_group,2)
        # ----------File path textedit setup-----------
        self.textEdit = textEdit_class(self.centralwidget,"textEdit_save",150, path3, 400, 30)
        self.textEdit_cam1 = textEdit_class(self.centralwidget,"textEdit_save",150, path1, 400, 30)
        self.textEdit_cam2 = textEdit_class(self.centralwidget,"textEdit_save",150, path2, 400, 30)

        # self.textEdit = QTextEdit(self.centralwidget)
        # self.textEdit.setGeometry(QtCore.QRect(150, 670, 400, 30))
        # self.textEdit.setFont(font)
        # self.textEdit.setObjectName("textEdit_save")
        #------------static rm checkbox-----------
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.sure_static = QtWidgets.QCheckBox(self.centralwidget)
        self.sure_static.setGeometry(QtCore.QRect(30, 600, 200, 30))
        self.sure_static.setFont(font)
        #------------------------------------------------
        grey = QPixmap(640, 480)
        grey.fill(QColor('darkGray'))
        self.image_label1 = QtWidgets.QLabel(self.centralwidget)
        self.image_label1.setPixmap(grey)
        self.image_label1.setGeometry(QtCore.QRect(1160, 30, 640, 480))
        self.image_label2 = QtWidgets.QLabel(self.centralwidget)
        self.image_label2.setPixmap(grey)
        self.image_label2.setGeometry(QtCore.QRect(1160,550, 640, 480))


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Real-Time Radar"))
        self.label.setText(_translate("MainWindow", "Range-Doppler Image"))
        self.label_2.setText(_translate("MainWindow", "Range-Angle Image"))
        self.label_pd.setText(_translate("MainWindow", "Point-cloud"))
        self.label_file.setText(_translate("MainWindow", "Path & Name:"))
        self.label_cam1.setText(_translate("MainWindow", "Cam1 Path:"))
        self.label_cam2.setText(_translate("MainWindow", "Cam2 Path:"))
        self.pd_btn.setText(_translate("MainWindow", "Point cloud"))
        self.load_btn.setText(_translate("MainWindow", "Load file"))
        self.browse_btn.setText(_translate("MainWindow", "Browse File"))
        self.browse_cam1_btn.setText(_translate("MainWindow", "Browse File"))
        self.browse_cam2_btn.setText(_translate("MainWindow", "Browse File"))
        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.stop_btn.setText(_translate("MainWindow", "Stop"))
        self.next_btn.setText(_translate("MainWindow", "Next frame"))
        self.pre_btn.setText(_translate("MainWindow", "Pre frame"))
        self.Cfar_rai.setText(_translate("MainWindow", "C-far"))
        self.beam_rai.setText(_translate("MainWindow", "beamforming"))
        self.orgin_rai.setText(_translate("MainWindow", "FFT-RAI"))
        self.sure_static.setText(_translate("MainWindow", "static_clutter_rm"))

