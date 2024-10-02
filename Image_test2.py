import math
import cv2
import numpy as np
import Preprocess

# Các hằng số
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
Min_char = 0.01
Max_char = 0.09
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Đọc và thay đổi kích thước hình ảnh
img = cv2.imread("data/image/20.jpg"
                 "")
img = cv2.resize(img, (1000, 1000))

# Tải mô hình KNN
npaClassifications = np.loadtxt("classifications.txt", np.float32).reshape(-1, 1)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Tiền xử lý hình ảnh
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tìm cạnh
dilated_image = cv2.dilate(canny_image, np.ones((3, 3), np.uint8), iterations=1)  # Giãn nở

# Tìm và lọc contour
contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contour lớn nhất

screenCnt = []
for c in contours:
    approx = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        screenCnt.append(approx)

if not screenCnt:
    print("No plate detected")
else:
    for cnt in screenCnt:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        # Tính toán góc và căn chỉnh biển số xe
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cnt[:, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = sorted(array, key=lambda x: x[1], reverse=True)
        doi, ke = abs(sorted_array[0][1] - sorted_array[1][1]), abs(sorted_array[0][0] - sorted_array[1][0])
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        # Cắt và căn chỉnh biển số
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [cnt], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (topx, topy), (bottomx, bottomy) = (np.min(x), np.min(y)), (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = ((bottomx - topx) / 2, (bottomy - topy) / 2)

        # Xoay và thay đổi kích thước biển số
        rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle if x1 < x2 else -angle, 1.0)
        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, None, fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, None, fx=3, fy=3)

        # Tiền xử lý và phân đoạn ký tự
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours_chars, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_x_ind = {}
        height, width, _ = roi.shape
        roiarea = height * width

        # Lọc ký tự
        for ind, cnt in enumerate(contours_chars):
            (x, y, w, h) = cv2.boundingRect(cnt)
            char_area = w * h
            ratiochar = w / h

            if Min_char * roiarea < char_area < Max_char * roiarea and 0.25 < ratiochar < 0.7:
                if x in char_x_ind:
                    x += 1
                char_x_ind[x] = ind

        # Nhận diện ký tự
        char_x = sorted(char_x_ind.keys())
        strFinalString = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(contours_chars[char_x_ind[i]])
            imgROI = thre_mor[y:y + h, x:x + w]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)).reshape(1, -1).astype(np.float32)

            _, npaResults, _, _ = kNearest.findNearest(imgROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))

            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
            strFinalString += strCurrentChar

        print(f"\nLicense Plate is: {strFinalString}\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow("ROI", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img)
cv2.waitKey(0)
