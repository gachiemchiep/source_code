from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Only needed for access to command line arguments
import sys


# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle("My Awesome App")
        

        layout = QVBoxLayout()
        widgets = [QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDial,
            QDoubleSpinBox,
            QFontComboBox,
            QLCDNumber,
            QLabel,
            QLineEdit,
            QProgressBar,
            QPushButton,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTimeEdit]
        
        for w in widgets:
            layout.addWidget(w())

        # Add some widget to layout
        widget = QLabel("Hello")
        font = widget.font()
        font.setPointSize(30)
        widget.setFont(font)
        widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)    
        
        layout.addWidget(widget)

        widget = QCheckBox()
        widget.setCheckState(Qt.Checked)
        widget.stateChanged.connect(self.show_state)

        layout.addWidget(widget)

        widget = QComboBox()
        widget.addItems(["One", "Two", "Three"])
        widget.currentIndexChanged.connect( self.index_changed )
        widget.currentIndexChanged[str].connect( self.text_changed )

        layout.addWidget(widget)

        widget = QListWidget()
        widget.addItems(["One_", "Two_", "Three_"])

        # In QListWidget there are two separate signals for the item, and the str
        widget.currentItemChanged.connect( self.index_changed )
        widget.currentTextChanged.connect( self.text_changed )
        layout.addWidget(widget)

        widget = QLineEdit()
        widget.setMaxLength(10)
        widget.setPlaceholderText("Enter your text")

        #widget.setReadOnly(True) # uncomment this to make readonly
        
        widget.returnPressed.connect(self.return_pressed)
        widget.selectionChanged.connect(self.selection_changed)
        widget.textChanged.connect(self.text_changed)
        widget.textEdited.connect(self.text_edited)

        layout.addWidget(widget)


        main_widget = QWidget()
        main_widget.setLayout(layout)
        
        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(main_widget)

    def show_state(self, s):
        print(s == Qt.Checked)
        print(s)

    def index_changed(self, i): # i is an int
        print(i)
        
    def text_changed(self, s): # s is a str
        print(s)

    def return_pressed(self):
        print("Return pressed!")
        self.centralWidget().setText("BOOM!")

    def selection_changed(self):
        print("Selection changed")
        print(self.centralWidget().selectedText())
        
    def text_changed(self, s):
        print("Text changed...")
        print(s)
            
    def text_edited(self, s):
        print("Text edited...")
        print(s)

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

window = MainWindow()
window.show() # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec_()


# Your application won't reach here until you exit and the event 
# loop has stopped.