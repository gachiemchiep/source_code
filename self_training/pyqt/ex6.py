import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import random 

SPRAY_PARTICLES = 100
SPRAY_DIAMETER = 10

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super(QtWidgets.QLabel, self).__init__()
        pixmap = QtGui.QPixmap(600, 300)
        self.setPixmap(pixmap)
        self.setStyleSheet("background-color: yellow;") 

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')    
        self.do_spray = False    

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def set_pen_mode(self, do_spray):
        self.do_spray = do_spray
        print("set_pen_mode: {}".format(do_spray))

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        if not self.do_spray:
            p.setWidth(4)
            p.setColor(self.pen_color)
            painter.setPen(p)
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            painter.end()
            self.update()

        else:
            p.setWidth(4)
            p.setColor(self.pen_color)
            painter.setPen(p)
            for n in range(SPRAY_PARTICLES):
                xo = random.gauss(0, SPRAY_DIAMETER)
                yo = random.gauss(0, SPRAY_DIAMETER)
                painter.drawPoint(e.x() + xo, e.y() + yo)

        print(self.do_spray)

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

COLORS = [
# 17 undertones https://lospec.com/palette-list/17undertones
'#000000', '#141923', '#414168', '#3a7fa7', '#35e3e3', '#8fd970', '#5ebb49', 
'#458352', '#dcd37b', '#fffee5', '#ffd035', '#cc9245', '#a15c3e', '#a42f3b', 
'#f45b7a', '#c24998', '#81588d', '#bcb0c2', '#ffffff',
]


class QPaletteButton(QtWidgets.QPushButton):

    def __init__(self, color):
        super(QtWidgets.QPushButton, self).__init__()
        self.setFixedSize(QtCore.QSize(24,24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()

        self.canvas = Canvas()

        main_widget = QtWidgets.QWidget()
        main_widget_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_widget_layout)

        # add toggle button here
        tool_widget = QtWidgets.QWidget()
        tool_layout = QtWidgets.QHBoxLayout()
        tool_widget.setLayout(tool_layout)
        b1 = QCheckBox("Spray?")
        b1.setChecked(True)
        b1.stateChanged.connect(lambda:self.canvas.set_pen_mode(b1.isChecked()))
        tool_layout.addWidget(b1)

        main_widget_layout.addWidget(self.canvas)

        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)
        main_widget_layout.addLayout(palette)
        main_widget_layout.addWidget(tool_widget)

        self.setCentralWidget(main_widget)

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()