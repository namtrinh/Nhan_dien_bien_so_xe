# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowState(QtCore.Qt.WindowMaximized)  # Open in full screen mode
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Vertical layout for the main window
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # Title label
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setText("XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH")
        self.layout.addWidget(self.label_title)

        # Subtitle label
        self.label_subtitle = QtWidgets.QLabel(self.centralwidget)
        font.setPointSize(24)
        self.label_subtitle.setFont(font)
        self.label_subtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_subtitle.setText("TÊN ĐỀ TÀI: NHẬN DIỆN BIỂN SỐ XE")
        self.layout.addWidget(self.label_subtitle)

        # Horizontal layout for image and text display
        self.content_layout = QtWidgets.QHBoxLayout()

        # Image label
        self.img = QtWidgets.QLabel(self.centralwidget)
        self.img.setAlignment(QtCore.Qt.AlignCenter)  # Center the image in the label
        self.img.setScaledContents(False)  # Disable scaling contents directly
        self.img.setMinimumSize(400, 300)  # Set minimum size for the image
        self.img.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # Allow image to resize but keep aspect ratio
        self.content_layout.addWidget(self.img)

        # Text edit for displaying information
        self.txt_img = QtWidgets.QTextEdit(self.centralwidget)
        self.txt_img.setPlaceholderText("Hiển thị thông tin biển số tại đây...")
        font_txt = QtGui.QFont()
        font_txt.setPointSize(14)  # Increase font size for better readability
        self.txt_img.setFont(font_txt)
        self.txt_img.setMinimumWidth(600)  # Make text area larger
        self.content_layout.addWidget(self.txt_img)

        # Add content layout to main layout
        self.layout.addLayout(self.content_layout)

        # Horizontal layout for buttons
        self.button_layout = QtWidgets.QHBoxLayout()

        # Define buttons
        self.btn_img = QtWidgets.QPushButton("IMAGE", self.centralwidget)
        self.btn_vid = QtWidgets.QPushButton("VIDEO", self.centralwidget)
        self.btn_img_detec = QtWidgets.QPushButton("IMAGE_DETECTION", self.centralwidget)
        self.btn_vid_detec = QtWidgets.QPushButton("VIDEO_DETECTION", self.centralwidget)


        # Set font and padding for larger buttons
        font_buttons = QtGui.QFont()
        font_buttons.setPointSize(16)  # Increase font size for buttons
        for button in [self.btn_img, self.btn_vid, self.btn_img_detec, self.btn_vid_detec]:
            button.setFont(font_buttons)
            button.setMinimumHeight(50)  # Increase button height for better appearance
            button.setMinimumWidth(150)  # Increase button width

        # Add buttons to horizontal layout
        self.button_layout.addWidget(self.btn_img)
        self.button_layout.addWidget(self.btn_vid)
        self.button_layout.addWidget(self.btn_img_detec)
        self.button_layout.addWidget(self.btn_vid_detec)


        # Add button layout to main layout
        self.layout.addLayout(self.button_layout)

        # Set layout for central widget
        self.centralwidget.setLayout(self.layout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Connect buttons to their respective functions
        self.btn_img.clicked.connect(self.load_image)
        self.btn_vid.clicked.connect(self.load_video)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH"))

    def load_image(self):
        # Clear the current image and text before loading new image
        self.clear_results()

        # Load new image (example path)
        image_path = "../.designer/backup/results.jpg"
        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            self.img.setPixmap(pixmap)
        else:
            self.txt_img.setText("Không thể tải hình ảnh.")

        # Display new information in the text area
        self.txt_img.setText("Thông tin biển số xe")

    def load_video(self):
        # Clear the current image and text before loading new video
        self.clear_results()

        # Placeholder text for video (as an example)
        self.txt_img.setText("Đang xử lý video...")

    def clear_results(self):
        # Clear the image and text fields
        self.img.clear()
        self.txt_img.clear()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()  # Show the window maximized
    sys.exit(app.exec_())
