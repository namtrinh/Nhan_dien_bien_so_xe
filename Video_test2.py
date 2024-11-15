import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import Preprocess  # Make sure to have Preprocess.py for preprocessing functions

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
Min_char = 0.01
Max_char = 0.09
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")
        self.root.geometry("1200x800")

        self.img_label = Label(root)
        self.img_label.pack()

        self.result_label = Label(root, text="Detected License Plate: ", font=("Helvetica", 16))
        self.result_label.pack()

        load_button = Button(root, text="Load Image", command=self.load_image)
        load_button.pack()

        self.kNearest = self.load_knn_model()

    def load_knn_model(self):
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return kNearest

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.resize(img, dsize=(800, 600))
            self.process_image(img)

    def process_image(self, img):
        imgGrayscale, imgThresh = Preprocess.preprocess(img)
        canny_image = cv2.Canny(imgThresh, 200, 255)
        kernel = np.ones((2, 2), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            self.result_label.config(text="No license plate detected")
            return

        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        mask = np.zeros(imgGrayscale.shape, np.uint8)
        cv2.drawContours(mask, [screenCnt], 0, 255, thickness=cv2.FILLED)
        y_coords, x_coords = np.where(mask == 255)
        topx, topy = np.min(y_coords), np.min(x_coords)
        bottomx, bottomy = np.max(y_coords), np.max(x_coords)

        roi = img[topx:bottomx, topy:bottomy]
        imgThreshPlate = imgThresh[topx:bottomx, topy:bottomy]

        ptPlateCenter = ((bottomx - topx) / 2, (bottomy - topy) / 2)
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        aligned_roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        aligned_imgThresh = cv2.warpAffine(imgThreshPlate, rotationMatrix, (bottomy - topy, bottomx - topx))

        aligned_roi = cv2.resize(aligned_roi, (0, 0), fx=4, fy=4)
        aligned_imgThresh = cv2.resize(aligned_imgThresh, (0, 0), fx=4, fy=4)

        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(aligned_imgThresh, cv2.MORPH_CLOSE, kerel3)
        thre_mor = cv2.morphologyEx(thre_mor, cv2.MORPH_DILATE, kerel3)

        cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = aligned_roi.shape
        roiarea = height * width
        char_x = []
        char_x_ind = {}

        for ind, cnt in enumerate(cont):
            x, y, w, h = cv2.boundingRect(cnt)
            char_area = w * h
            if Min_char * roiarea < char_area < Max_char * roiarea and 0.2 < w / h < 0.8:
                char_x.append(x)
                char_x_ind[x] = ind

        char_x = sorted(char_x)
        first_line = ""
        second_line = ""

        for i in char_x:
            x, y, w, h = cv2.boundingRect(cont[char_x_ind[i]])
            imgROI = thre_mor[y:y + h, x:x + w]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            npaROIResized = np.float32(npaROIResized)

            _, npaResults, _, _ = self.kNearest.findNearest(npaROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))

            if y < height / 4:
                first_line += strCurrentChar
            else:
                second_line += strCurrentChar

        license_plate = first_line + "-" + second_line
        self.result_label.config(text="Detected License Plate: " + license_plate)

        self.display_image(img)

    def display_image(self, img):
        img = cv2.resize(img, (600, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=im_pil)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
