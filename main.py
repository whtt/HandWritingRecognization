# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  : the main gui for the project
@Project: PR_HW
@File   : main.py
@Author : whtt
@Time   : 2020/10/5 13:32
"""

from PyQt5.Qt import QMainWindow, QWidget, QColor, QPixmap, QIcon, QSize, QFrame, \
    QCheckBox, QLineEdit, QAction, QMessageBox, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, \
    QComboBox, QLabel, QSpinBox, QFileDialog, QApplication
from PaintBoard import PaintBoard
from utils import create_dir_path, create_file_path, Logger
from kernel import Kernel
import sys
import os
from PyQt5 import QtCore


class MyWidget(QWidget):
    log = Logger('logs/main.log', level='info')
    my_kernel = Kernel(log.logger)

    def __init__(self, Parant=None):
        super().__init__(Parant)
        self.__eraser_mode_set = False
        self.__init_data()
        self.__init_view()

    def __init_data(self):
        self.__canvas = PaintBoard(self)
        self.__colorList = QColor.colorNames()

    def __init_view(self):
        self.setFixedSize(720, 530)

        self.label_university = QLabel("学校：\t华南理工大学", self)
        self.label_university.setStyleSheet("font-size:12px")
        self.label_university.setGeometry(460, 10, 120, 35)

        self.label_school = QLabel("专业：\t控制科学与工程", self)
        self.label_school.setStyleSheet("font-size:12px")
        self.label_school.setGeometry(460, 40, 140, 35)

        self.label_name = QLabel("姓名：\t汪皓", self)
        self.label_name.setStyleSheet("font-size:12px")
        self.label_name.setGeometry(460, 70, 100, 35)

        self.label_number = QLabel("学号：\t202020116491", self)
        self.label_number.setStyleSheet("font-size:12px")
        self.label_number.setGeometry(460, 100, 120, 35)

        self.label_im = QLabel(self)
        self.label_im.setGeometry(600, 20, 105, 105)
        self.label_im.setPixmap(QPixmap("./im_scut.jpg").scaled(105, 105))

        self.label_log = QTextEdit(self)
        self.label_log.setReadOnly(True)
        self.label_log.setStyleSheet("background:transparent; font-size:15px; font-family:Roman times")
        # self.label_log.resize(220, 200)
        self.label_log.setGeometry(460, 140, 245, 220)
        self.label_log.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label_log.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # create a horizontal layout as the main layout of the GUI
        main_layout = QHBoxLayout(self)
        # set the distance between widgets to 10px
        main_layout.setSpacing(10)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)

        left_layout.addWidget(self.__canvas)

        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(5, 5, 5, 5)

        self.__lab_path = QLabel(self)
        self.__lab_path.setText("Save Path")
        self.__lab_path.setFixedHeight(10)
        path_layout.addWidget(self.__lab_path)
        left_layout.addLayout(path_layout)

        self.__txt_path = QLineEdit("./data/1")
        path_layout.addWidget(self.__txt_path)

        self.__choose_path = QPushButton("index")
        self.__choose_path.setParent(self)
        # self.__choose_path.setShortcut("Ctrl+S")
        self.__choose_path.clicked.connect(self.btn_save_clicked)
        path_layout.addWidget(self.__choose_path)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(5, 5, 5, 5)

        self.__btn_save = QPushButton("Save")
        self.__btn_save.setParent(self)
        self.__btn_save.setShortcut("Ctrl+A")
        self.__btn_save.clicked.connect(self.save_image)
        btn_layout.addWidget(self.__btn_save)

        self.__btn_recognize = QPushButton("Predict")
        self.__btn_recognize.setParent(self)
        self.__btn_recognize.setShortcut("Ctrl+R")
        self.__btn_recognize.clicked.connect(self.btn_recog_clicked)
        btn_layout.addWidget(self.__btn_recognize)

        self.__btn_clear = QPushButton("Clear")
        self.__btn_clear.setShortcut("Ctrl+X")
        self.__btn_clear.setParent(self)
        self.__btn_clear.clicked.connect(self.btn_clear_clicked)
        btn_layout.addWidget(self.__btn_clear)

        self.__btn_exit = QPushButton("EXIT")
        self.__btn_exit.setParent(self)
        # self.__btn_exit.setShortcut("Ctrl+Q")
        self.__btn_exit.clicked.connect(self.btn_exit_clicked)
        btn_layout.addWidget(self.__btn_exit)

        left_layout.addLayout(btn_layout)
        main_layout.addLayout(left_layout)

        # put canvas in the left GUI
        # main_layout.addWidget(self.__canvas)
        # main_layout.addWidget(left_layout)

        # set the right layout
        right_layout = QVBoxLayout()
        # set the space between right contents
        right_layout.setContentsMargins(5, 5, 5, 14)

        splitter = QSplitter(self)
        right_layout.addWidget(splitter)

        method_layout = QHBoxLayout()

        self.__lab_method = QLabel(self)
        self.__lab_method.setText("Recognition Method")
        self.__lab_method.setFixedHeight(30)
        method_layout.addWidget(self.__lab_method)

        self.__box_method = QComboBox(self)
        self.__box_method.addItems(['NaiveBayesian', 'Fisher', 'SVM', 'VGG16bn'])
        self.__box_method.setCurrentIndex(0)
        method_layout.addWidget(self.__box_method)

        right_layout.addLayout(method_layout)

        self.__cbtn_eraser = QCheckBox("Eraser")
        self.__cbtn_eraser.setParent(self)
        self.__cbtn_eraser.clicked.connect(self.btn_eraser_clicked)
        right_layout.addWidget(self.__cbtn_eraser)

        pen_size_layout = QHBoxLayout()

        self.__lab_pen_size = QLabel(self)
        self.__lab_pen_size.setText("Pen Size")
        self.__lab_pen_size.setFixedHeight(30)
        pen_size_layout.addWidget(self.__lab_pen_size)

        self.__box_pen_size = QSpinBox(self)
        self.__box_pen_size.setMaximum(40)
        self.__box_pen_size.setMinimum(2)
        self.__box_pen_size.setValue(20)
        self.__box_pen_size.setSingleStep(2)
        self.__box_pen_size.valueChanged.connect(self.box_pen_size_change)
        pen_size_layout.addWidget(self.__box_pen_size)
        right_layout.addLayout(pen_size_layout)

        pen_color_layout = QHBoxLayout()

        self.__label_pen_color = QLabel(self)
        self.__label_pen_color.setText("Pen Color")
        self.__label_pen_color.setFixedHeight(30)
        pen_color_layout.addWidget(self.__label_pen_color)

        self.__combo_pen_color = QComboBox(self)
        self.__fillColorList(self.__combo_pen_color)
        self.__combo_pen_color.currentIndexChanged.connect(self.pen_color_changed)
        pen_color_layout.addWidget(self.__combo_pen_color)
        right_layout.addLayout(pen_color_layout)

        main_layout.addLayout(right_layout)

    def btn_recog_clicked(self):
        savePath = "./recog.jpg"
        image = self.__canvas.get_current_image()
        image.save(savePath)
        save_path = os.path.abspath(savePath)
        self.label_log.append("image saved in path:\n{}".format(save_path))
        method_text = self.__box_method.currentText()
        method = Kernel.set_kernel(method_text)
        predict = method.predict(savePath)
        self.label_log.append("recognition result is: {}".format(predict))
        message = QMessageBox()
        message.setText("recognition result is: {}".format(predict))
        # message.addButton()
        message.exec_()

    def btn_clear_clicked(self):
        self.__canvas.clear()
        self.label_log.append("Canvas is clean now!")

    def btn_save_clicked(self):
        save_path = QFileDialog.getSaveFileName(self, 'save your paint', '.\\', '*.jpg')
        if save_path[0] == "":
            self.label_log.append("save cancel")
            return
        self.__txt_path.setText(save_path[0])
        save_image = self.__canvas.get_current_image()
        save_image.save(save_path[0])
        save_path_ = os.path.abspath(save_path[0])
        self.label_log.append("image saved in path:\n{}".format(save_path_))

    def save_image(self):
        self.__txt_path.setText(self.__txt_path.displayText())
        save_path = self.__txt_path.text()
        created = create_dir_path(save_path)
        if created:
            self.label_log.append("create an directory:\n{}".format(save_path))
        file_path = create_file_path(save_path)
        save_image = self.__canvas.get_current_image()
        save_image.save(file_path)
        file_path_ = os.path.abspath(file_path)
        self.label_log.append("image saved in path:\n{}".format(file_path_))

    def btn_exit_clicked(self):
        self.close()
        QApplication.quit()

    def btn_eraser_clicked(self):
        self.__eraser_mode_set = ~self.__eraser_mode_set
        if self.__eraser_mode_set:
            self.label_log.append("Attention! Eraser Mode!")
        else:
            self.label_log.append("Writing Mode!")
        if self.__cbtn_eraser.isChecked():
            self.__canvas.EraserMode = True
        else:
            self.__canvas.EraserMode = False

    def box_pen_size_change(self):
        pen_size = self.__box_pen_size.value()
        self.__canvas.pen_size(pen_size)

    def __fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def pen_color_changed(self):
        color_index = self.__combo_pen_color.currentIndex()
        color_str = self.__colorList[color_index]
        self.__canvas.pen_color(color_str)


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("Pattern Recognition")
        self.setStyleSheet("font-size:14px; font-family:Roman Times")

        self.__widget = widget

        self.__menu = self.menuBar()
        self.__file_menu = self.__menu.addMenu("File")
        self.__view_menu = self.__menu.addMenu("View")

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_app)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.__file_menu.addAction(save_action)
        self.__file_menu.addAction(exit_action)

        clear_log_action = QAction("Log_Clear", self)
        clear_log_action.setShortcut("Ctrl+E")
        clear_log_action.triggered.connect(self.clear_log_app)

        self.__view_menu.addAction(clear_log_action)

        self.setCentralWidget(widget)

    def exit_app(self):
        QApplication.quit()

    def save_app(self):
        self.__widget.btn_save_clicked()

    def clear_log_app(self):
        self.__widget.label_log.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    widget = MyWidget()

    window = MainWindow(widget)
    window.resize(720, 550)
    window.show()

    exit(app.exec_())

