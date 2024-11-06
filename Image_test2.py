import math
import cv2
import numpy as np
import Preprocess

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
n = 1
Min_char = 0.01
Max_char = 0.09
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

img = cv2.imread("data/image/img_13.png")
img = cv2.resize(img, dsize=(1920, 1080))

######## Tải mô hình KNN ######################
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # Định hình lại mảng numpy thành 1d, cần thiết để truyền cho hàm train
kNearest = cv2.ml.KNearest_create()  # Khởi tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

################ Xử lý ảnh #################
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 200, 255)  # Cạnh Canny
kernel = np.ones((2, 2), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=2)  # Giãn nở
cv2.imshow("dilated_image",dilated_image)

###########################################

###### Vẽ contour và lọc ra biển số xe #############
#ìm tất cả các contour trong ảnh giãn nở
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]  # Lấy 10 contours có diện tích lớn nhất

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    #trả về hình chữ nhật bao quanh approx
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    if (len(approx) == 4):
        screenCnt.append(approx)

if screenCnt is None:
    detected = 0
    print("Không phát hiện biển số")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Tìm góc của biển số xe #####################
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

        ########## Cắt biển số xe và căn chỉnh nó theo góc đúng ################

        # Tạo mặt nạ với kích thước giống ảnh gốc, dùng để xác định vùng biển số
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)

        # Vẽ contour của biển số lên mặt nạ, tô đầy vùng biển số bằng giá trị 255 (màu trắng)
        cv2.drawContours(mask, [screenCnt], 0, 255, thickness=cv2.FILLED)

        # Lấy tọa độ các điểm thuộc vùng biển số
        (y_coords, x_coords) = np.where(mask == 255)

        # Tính toán các toạ độ cắt chính xác dựa trên tọa độ tìm được
        topx, topy = np.min(y_coords), np.min(x_coords)
        bottomx, bottomy = np.max(y_coords), np.max(x_coords)

        # Cắt ảnh vùng biển số từ ảnh gốc và ảnh nhị phân
        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]

        # Tính toán trung tâm của vùng biển số để căn chỉnh xoay
        ptPlateCenter = ((bottomx - topx) / 2, (bottomy - topy) / 2)

        # Xác định góc xoay và thực hiện phép xoay để biển số được căn chỉnh đúng
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        # Căn chỉnh vùng biển số về góc ngang
        aligned_roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        aligned_imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

        # Tăng kích thước vùng biển số để dễ xử lý hơn
        aligned_roi = cv2.resize(aligned_roi, (0, 0), fx=4, fy=4)
        aligned_imgThresh = cv2.resize(aligned_imgThresh, (0, 0), fx=4, fy=4)

        #################### Tiền xử lý và phân đoạn ký tự ####################
        # Tạo phần tử cấu trúc hình chữ nhật 3x3 để sử dụng trong các phép biến đổi hình thái học
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Áp dụng phép giãn nở để tăng độ nổi bật của ký tự
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kerel3)
        thre_mor = cv2.morphologyEx(thre_mor, cv2.MORPH_DILATE, kerel3)

        # Tìm các contour trên ảnh đã qua xử lý
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hiển thị ảnh đã qua xử lý
        #cv2.imshow(str(n + 20), thre_mor)

        # Vẽ các contour tìm được trên vùng chứa biển số
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các ký tự trong biển số

        ##################### Lọc ký tự #################
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        # Duyệt qua các contour và lọc các ký tự phù hợp
        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cnt)
            ratiochar = w / h
            char_area = w * h
            # Điều kiện lọc ký tự dựa vào diện tích và tỷ lệ ký tự
            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.2 < ratiochar < 0.8):
                if cv2.contourArea(cnt) > 10:  # Điều kiện này có thể điều chỉnh tùy theo kích thước thực tế của ký tự
                    char_x.append(x)
                    char_x_ind[x] = ind
        # Nhận dạng ký tự
        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""
        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])

            # Kiểm tra lại các điều kiện lọc để chắc chắn rằng ký tự được chọn là đúng
            if w > 0 and h > 0:
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ khung quanh ký tự

                # Cắt ký tự từ ảnh đã qua xử lý
                imgROI = thre_mor[y:y + h, x:x + w]

                # Thay đổi kích thước ảnh ký tự để chuẩn hóa kích thước
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaROIResized = np.float32(npaROIResized)

                # Nhận dạng ký tự bằng KNN
                _, npaResults, _, _ = kNearest.findNearest(npaROIResized, k=3)
                strCurrentChar = str(chr(int(npaResults[0][0])))  # Chuyển đổi kết quả về ký tự ASCII

                # Phân chia ký tự vào dòng đầu hoặc dòng thứ hai dựa trên tọa độ y
                if (y < height / 4):  # quyết định biển số 1 dòng hay 2 dòng
                    first_line += strCurrentChar
                else:
                    second_line += strCurrentChar
        print("\n Biển số là: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow("Biển số đã nhận dạng", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Hiển thị biển số đã nhận dạng trên ảnh gốc
        cv2.putText(img, first_line + "-" + second_line, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Biển số xe', img)

cv2.waitKey(0)
