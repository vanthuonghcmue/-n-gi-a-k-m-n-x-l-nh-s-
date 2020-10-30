import argparse

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QGraphicsDropShadowEffect
import sys
import cv2,imutils
from math import *
from matplotlib import pyplot as plt
import numpy as np
import copy

# ==> GLOBALS
counter = 0


# YOUR APPLICATION
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        uic.loadUi('2D.ui', self)
        self.beforImage.setPixmap(QtGui.QPixmap('meoo.jpg'))
        self.affterImage.setPixmap(QtGui.QPixmap('test.png'))
        self.imglog.setPixmap(QtGui.QPixmap('1.png'))
        self.Bilateralx.setPixmap(QtGui.QPixmap('2.png'))
        self.tb_mean_21.setPixmap(QtGui.QPixmap('3.png'))
        self.tb_mean_22.setPixmap(QtGui.QPixmap('4.png'))
        self.tb_mean_24.setPixmap(QtGui.QPixmap('5.png'))
        self.tb_mean_29.setPixmap(QtGui.QPixmap('6.png'))
        self.tb_mean_28.setPixmap(QtGui.QPixmap('7.png'))
        self.tb_mean_27.setPixmap(QtGui.QPixmap('8.png'))
        self.tb_mean_26.setPixmap(QtGui.QPixmap('9.png'))
        self.tb_K_Mean.setPixmap(QtGui.QPixmap('10.png'))
        self.histogramimg_2.setPixmap(QtGui.QPixmap('11.png'))
        self.dsda.setPixmap(QtGui.QPixmap('12.png'))
        self.GXds.setPixmap(QtGui.QPixmap('13.png'))
        self.dsda_4GY.setPixmap(QtGui.QPixmap('14.png'))
        self.GxAddGysa_2.setPixmap(QtGui.QPixmap('15.png'))
        self.images = None
        self.origin = None
        self.Gray = None
        self.szFilter = None
        self.addImage.clicked.connect(self.openFile)
        self.mean.clicked.connect(self.Mean)
        self.blur.clicked.connect(self.Blur)
        self.gauss.clicked.connect(self.Gauss)
        self.median.clicked.connect(self.Median)
        self.Segmentation.clicked.connect(self.thresholding)
        self.otsu.clicked.connect(self.Otsu)
        self.Segmentation2.clicked.connect(self.AdaptiveThreshold)
        self.KMean.clicked.connect(self.K_Mean)
        self.addImage_2.clicked.connect(self.writeImage)
        self.log.clicked.connect(self.Log)
        self.histogram.clicked.connect(self.Histogram)
        self.bilateral.clicked.connect(self.Bilateral)
        self.directional.clicked.connect(self.Directional)
        self.GX.clicked.connect(self.Gx)
        self.GY.clicked.connect(self.Gy)
        self.GxAddGy.clicked.connect(self.GxAddGyy)





    def openFile(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.origin = self.image
        self.Gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.setPhoto(self.image, self.beforImage)

    def setPhoto(self, image, label):
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(image))



    def Exit(self):
        quit(0)

    def ShowImageGray(self, image, label):
        label.clear()
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             QtGui.QImage.Format_Grayscale8).rgbSwapped()
        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def Mean(self):
        # self.labelAfter.setText('AFTER APPLY MEAN FILTER')
        self.szFilter = self.spinBox.value()
        img = copy.copy(self.origin)
        img_filter = np.ones(shape=(self.szFilter, self.szFilter))
        img_filter = img_filter / sum(img_filter)
        self.images = cv2.filter2D(img, -1, img_filter)
        self.setPhoto(self.images, self.affterImage)

    def Blur(self):
        # self.labelAfter.setText('AFTER APPLY BLUR FILTER')
        self.szFilter = self.spinBox.value()
        # img = copy.copy(self.Gray)
        self.images = cv2.blur(self.origin, (self.szFilter, self.szFilter))
        self.setPhoto(self.images, self.affterImage)

    def Gauss(self):
        # self.labelAfter.setText('AFTER APPLY GAUSS FILTER')
        self.szFilter = self.spinBox.value()
        img = copy.copy(self.Gray)
        self.images = cv2.GaussianBlur(img, (self.szFilter, self.szFilter),0)
        self.ShowImageGray(self.images, self.affterImage)

    def Median(self):
        # self.labelAfter.setText('AFTER APPLY MEDIAN FILTER')
        self.szFilter = self.spinBox.value()
        img = copy.copy(self.origin)
        self.images = cv2.medianBlur(img, self.szFilter)
        self.setPhoto(self.images, self.affterImage)

    def thresholding (self):
        img = copy.copy(self.Gray)
        img = cv2.medianBlur(img, 5)

        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        self.images = th2
        self.ShowImageGray(self.images, self.affterImage)

    def Otsu (self):
        img = copy.copy(self.Gray)
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.images =  th2
        self.ShowImageGray(self.images, self.affterImage)

    def AdaptiveThreshold(self, cv=None):
        img = copy.copy(self.Gray)
        self.szFilter = self.spinBox.value()
        img = cv2.medianBlur(img, 5)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,  self.szFilter, 2)
        self.images = th3
        self.ShowImageGray(self.images, self.affterImage)

    def K_Mean(self):
        img = self.origin
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        self.images = res.reshape((img.shape))
        self.setPhoto(self.images, self.affterImage)

    # Apply log and gamma
    def Log(self):
        # Apply gamma correction.
        img = self.origin
        self.szFilter = self.spinBox.value()
        self.images = np.array(255 * (img / 255) ** self.szFilter, dtype='uint8')
        self.setPhoto(self.images, self.affterImage)

    # histogram
    def Histogram(self):
        img = self.origin
        parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
        parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
        args = parser.parse_args()
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.images = cv2.equalizeHist(src)
        self.setPhoto(self.images, self.affterImage)

    # directional
    def Directional(self):
        img = copy.copy(self.Gray)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        self.images = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        self.ShowImageGray(self.images, self.affterImage)

    # median
    def Bilateral (self):
        img = self.origin
        self.szFilter = self.spinBox.value()
        self.images = cv2.bilateralFilter(img,9,75,75)
        self.setPhoto(self.images, self.affterImage)

    #Gradient
    def getGx(self):
        img = copy.copy(self.Gray)
        filterGx = np.array([
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0]
        ])
        img_Gx = cv2.filter2D(img, -1, filterGx)
        return img_Gx
        # test = img_Gx + self.getGy()
        # self.ShowImageGray(img_Gx, self.iframe_new)

    def getGy(self):
        img = copy.copy(self.Gray)
        filterGy = np.array([
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0]
        ])
        img_Gy = cv2.filter2D(img, -1, filterGy)
        return img_Gy

    def Gx(self):

        self.images = self.Gray - self.getGx()
        self.ShowImageGray(self.images, self.affterImage)


    def Gy(self):
       self.images = self.getGy()
       self.ShowImageGray(self.images, self.affterImage)

    def GxAddGyy(self):
        self.images = self.getGx() + self.getGy()
        self.ShowImageGray(self.images, self.affterImage)

    # save image
    def writeImage (self):
        result = cv2.imwrite(r'C:\Users\DELL\PycharmProjects\imageFilter\hinhmoi.png', self.images)
        if result == True:
            self.save.setText('save in downloads')
        else:
            self.save.setText('do not save')


# SPLASH SCREEN
class SPLASH(QtWidgets.QMainWindow):
    def __init__(self):
        super(SPLASH, self).__init__()
        uic.loadUi('flash_screen.ui', self)

        ## UI-->  INTERFACE CODE
        ############################################################

        ## REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


        ## DROP SHADOW EFFRCT

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,0,0,60))
        self.dropshawdowframe.setGraphicsEffect(self.shadow)

        ## QTIME ==> STRART
        self.timer = QtCore .QTimer()
        self.timer.timeout.connect(self.progress)
        #TIME IN MILLISECONDS
        self.timer.start(35)

        # CHANG DESCRIPTION

        # Initial text
        self.description_2.setText("<strong>WELCOM</strong> TO MY APPLICATION")

        # Change Text
        QtCore.QTimer.singleShot(1500, lambda: self.description_2.setText("<strong>LOADING</strong> DATABASE"))
        QtCore.QTimer.singleShot(3000, lambda: self.description_2.setText("<strong>LOADING</strong> USE INTERFACE"))



        # SHOW ==> MAIN WINDOW
        self.show()
        ##  ==> END ##


    def progress(self):

        global counter

        ## SET VALUE TO PROGRESS BAR
        self.progressBar.setValue(counter)

        ## CLOSE SPLASH SCREE AND OPEN APP
        if counter > 150:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = Ui_MainWindow()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        #INCREASE COUNTER
        counter += 1


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SPLASH()
    app.exec_()
