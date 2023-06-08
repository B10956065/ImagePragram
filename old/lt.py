import cv2
import numpy as np

# 讀取影像
image = cv2.imread('BottlenoseDolphins.png')

# 定義 Beta 值
beta = 50

# 將原始影像轉換為浮點型數據類型
image_float64 = image.astype(np.float64) / 255.0  # 正規化到 [0, 1] 範圍

# 進行 Beta 校正
beta_corrected = cv2.add(image_float64, np.array([beta, beta, beta]) / 255.0)  # 正規化 Beta 值

# 將結果轉換為無符號 8 位整數型數據類型
beta_corrected = np.uint8(beta_corrected * 255)  # 將結果還原到 [0, 255] 範圍

# 顯示原始影像和 Beta 校正後的影像
cv2.imshow('Original Image', image)
cv2.imshow('Beta Corrected Image', beta_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()
