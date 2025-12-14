import os
import cv2 as cv

# Проверяем существование файла
picture_files_path = './jpg/'
file1 = picture_files_path + 'dict__.jpg'


dict_img = cv.imread(file1)
    
if dict_img is None:
    print("Ошибка загрузки изображения")
else:
    print(f"Изображение загружено успешно. Размер: {dict_img.shape}")
    cv.imshow('image', dict_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


blurred = cv.GaussianBlur(dict_img, (5, 5), 0)
cv.GaussianBlur(dict_img, (5,5), 0)
cv.imshow('blur', blurred)
cv.waitKey(0)
cv.destroyAllWindows()



dict_img = cv.imread(file1)
dict_hsv = cv.blur(dict_img, (2, 2))
cv.imshow('blur', blurred)
cv.waitKey(0)
cv.destroyAllWindows()
dict_hsv = cv.cvtColor(dict_hsv, cv.COLOR_BGR2HSV)
cv.imshow('blur', blurred)
cv.waitKey(0)
cv.destroyAllWindows()
dict_mask = cv.inRange(dict_hsv, (0, 0, 0), (255, 255, 141))


cv.imshow('dict2', dict_mask)
cv.waitKey(0)
cv.destroyAllWindows()