# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  :
@Project: PR_HW
@File   : PaintBoard.py
@Author : whtt
@Time   : 2020/10/5 13:32 14:26
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__init_data()
        self.__init_view()

    def __init_data(self):
        """
        initial the canvas, painter
        :return:
        """
        self.__size = QSize(420, 420)

        # create a new canvas by QPixmap, size for self.__size
        self.__canvas = QPixmap(self.__size)
        # set the background of the board as white, for better visual effect
        self.__canvas.fill(Qt.white)

        # default for none
        self.__IsEmpty = True
        # default for no eraser
        self.EraserMode = False

        # initial the last mouse position
        self.__lastPos = QPoint(0, 0)
        # initial the current mouse position
        self.__currentPos = QPoint(0, 0)

        # new a painter for drawing
        self.__painter = QPainter()

        # default pen size for 10px
        self.__thickness = 20
        # default pen color for black
        self.__penColor = QColor("black")
        # get the color list from library
        self.colorList = QColor.colorNames()

    def __init_view(self):
        """
        set the initial size of the canvas
        :return:
        """
        self.setFixedSize(self.__size)

    def clear(self):
        """
        clear the canvas
        :return:
        """
        self.__canvas.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def pen_color(self, color="black"):
        """
        set the color of the pen
        :param color:
        :return:
        """
        self.__penColor = QColor(color)

    def pen_size(self, thick=20):
        """
        set the size of the pen
        :param thick:
        :return:
        """
        self.__thickness = thick

    def is_empty(self):
        """
        return the canvas is empty or not
        :return:
        """
        return self.__IsEmpty

    def get_current_image(self):
        """
        fet the current content of the canvas, return as an image
        :return:
        """
        current_image = self.__canvas.toImage()
        return current_image

    def paintEvent(self, paintEvent):
        """
        the painter works between begin() and end()
        - begin(param): parameter--canvas
        - drawPixmap: paint QPixmap object
            0, 0 start
        :param paintEvent:
        :return:
        """
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__canvas)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        """
        capture the mouse when pressed
        :param mouseEvent:
        :return:
        """
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        """
        when the mouse moves, update the position
        :param mouseEvent:
        :return:
        """
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__canvas)

        if self.EraserMode == False:
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
        else:
            self.__painter.setPen((QPen(Qt.white, 20)))

        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        """
        set the canvas for not empty
        :param QMouseEvent:
        :return:
        """
        self.__IsEmpty = False
