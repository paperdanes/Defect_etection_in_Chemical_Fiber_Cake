"""
ed_detect.py

封装：读取单张图片 → 边缘检测（Canny） → Hough 检测内外圆

公开主函数：
    run_edge_detection(image_path: str) -> (img_bgr, img_gray, edges, circles)

circles 形状为 (N, 3)，每行 (x, y, r)，N 可以是 0/1/2。
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
from matplotlib import pyplot as plt


# ================== 内部工具函数 ================== #

def _load_image_as_gray(image_path: str):
    """
    读取 BGR 图像，并转换为灰度图。
    返回:
        img_bgr: 原图 (H, W, 3)
        img_gray: 灰度图 (H, W)
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    i_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, i_gray


def _detect_edges(i_gray: np.ndarray,
                  threshold1: int = 70,
                  threshold2: int = 80) -> np.ndarray:
    """
    在灰度图上做 Canny 边缘检测。

    返回:
        edges: 0/255 的边缘图 (H, W)
    """
    edge = cv2.Canny(i_gray, threshold1=threshold1, threshold2=threshold2)
    return edge


def _detect_outer_inner_circles(
    i_gray: np.ndarray,edge: np.ndarray,
    outer_radius_center: int = 1500,
    outer_radius_margin: int = 100,
    inner_radius_center: int = 800,
    inner_radius_margin: int = 50,
    dp: float = 1.2,param1: float = 80,param2: float = 40,) :
    """
    使用 HoughCircles 分别检测外圆和内圆。

    返回:
        circles: shape (N, 3) 的 int 数组，每行 (x, y, r)，N ∈ {0,1,2}
    """
    h, w = i_gray.shape[:2]
    gray_for_hough = i_gray  # 目前用灰度图作为 Hough 输入

    minDist = min(h, w) // 2   # 每次只想要一个圆，设大一点，避免重复

    # ----- 外圆：半径约 outer_radius_center ± outer_radius_margin -----
    outer_min_r = outer_radius_center - outer_radius_margin
    outer_max_r = outer_radius_center + outer_radius_margin

    circles_outer_raw = cv2.HoughCircles(
        gray_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=outer_min_r,
        maxRadius=outer_max_r,
    )

    outer_circle = None
    if circles_outer_raw is not None:
        cps_o = np.squeeze(circles_outer_raw, axis=0).astype(np.float32)  # (N,3)
        # 选半径最接近期望值的那个
        idx_o = np.argmin(np.abs(cps_o[:, 2] - outer_radius_center))
        outer_circle = tuple(cps_o[idx_o])

    # ----- 内圆：半径约 inner_radius_center ± inner_radius_margin -----
    inner_min_r = inner_radius_center - inner_radius_margin
    inner_max_r = inner_radius_center + inner_radius_margin

    circles_inner_raw = cv2.HoughCircles(
        gray_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=inner_min_r,
        maxRadius=inner_max_r,
    )

    inner_circle = None
    if circles_inner_raw is not None:
        cps_i = np.squeeze(circles_inner_raw, axis=0).astype(np.float32)
        idx_i = np.argmin(np.abs(cps_i[:, 2] - inner_radius_center))
        inner_circle = tuple(cps_i[idx_i])


    circles_list: List[Tuple[int, int, int]] = []
    if outer_circle is not None:
        xo, yo, ro = outer_circle
        circles_list.append((int(round(xo)), int(round(yo)), int(round(ro))))

    if inner_circle is not None:
        xi, yi, ri = inner_circle
        circles_list.append((int(round(xi)), int(round(yi)), int(round(ri))))

    if not circles_list:
        return np.empty((0, 3), dtype=np.int32)

    return np.asarray(circles_list, dtype=np.int32)


# ================== 对外主函数 ================== #

def run_edge_detection(image_path: str,canny_t1: int = 70,canny_t2: int = 80) :
    """
    对指定图片做内外圆的边缘检测。
    输入:
        image_path: 图片路径
        canny_t1, canny_t2: Canny 阈值
    输出:
        img_bgr  : 原始 BGR 图
        img_gray : 灰度图
        edges    : Canny 边缘图 (H, W)
        circles  : (N,3) 的数组，每行 (x, y, r)，N 可能为 0/1/2
    """
    Img, Img_gray = _load_image_as_gray(image_path)
    Edges = _detect_edges(Img_gray, threshold1=canny_t1, threshold2=canny_t2)
    Circles = _detect_outer_inner_circles(Img_gray, Edges)

    return Img,Img_gray, Edges, Circles


if __name__ == "__main__":

    test_path = os.path.join("stainimg","cam1_20230305195107998.bmp")

    print(f"测试图片路径: {test_path}")
    Img, Img_gray, Edges, Circles = run_edge_detection(test_path)

    print("图像尺寸 (H, W, C):",Img.shape)
    print("灰度尺寸 (H, W):",Img_gray.shape)
    print("检测到的圆 (x, y, r):")
    print(Circles)

    # 在原图上画圆
    img_with_circles = Img.copy()
    if Circles is not None and len(Circles) > 0:
        # 预设两个颜色：第一个红色，第二个绿色
        colors = [(255, 0, 0), (0, 255, 0), ]
        for idx, (x, y, r) in enumerate(Circles):
            if idx < 2:
                color = colors[idx]
            else:
                color = (0, 0, 255)
            cv2.circle(img_with_circles, (int(x),int(y)), int(r), color, 10)
            # 如果想把圆心也标出来：
            cv2.circle(img_with_circles, (x, y), 20, color, -1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original Img")
    plt.imshow(Img)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Gray Img")
    plt.imshow(Img_gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Edges (Canny)")
    plt.imshow(Edges, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Circles on Original Img")
    plt.imshow(img_with_circles)
    plt.axis("off")

    plt.tight_layout()
    plt.show()