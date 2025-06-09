import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转为灰度图
img_path = "/mnt/data/610baed1-19b5-474f-bbcb-04ce131632e0.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值处理：将非灰色区域（二值化）视为白色空间
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# 连通域标记：将白色区域划分为不同的空间并编号
num_labels, labels_im = cv2.connectedComponents(binary)

# 随机颜色映射以区分不同区域
label_hue = np.uint8(179 * labels_im / np.max(labels_im))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0  # 背景设为黑色

# 显示结果
plt.figure(figsize=(8, 8))
plt.imshow(labeled_img)
plt.title(f"白色空间区域划分（共 {num_labels - 1} 个区域）")
plt.axis("off")
plt.show()
