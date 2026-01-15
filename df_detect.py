from __future__ import annotations
import os
from typing import Tuple, Dict, List, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ed_detect import run_edge_detection

def build_ring_mask(gray_img: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """
    根据 Hough 检测到的圆，构造环形掩膜。
    circles: (N, 3)，每行 (x, y, r)，N 期望为 2（外圆 + 内圆）

    返回:
        mask (H, W), uint8，环形区域为 255，其余为 0
    """
    h, w = gray_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if circles is None or len(circles) == 0:
        raise RuntimeError("build_ring_mask: 没有检测到任何圆，无法构造环形掩膜。")

    # 按半径从大到小排序，最大视作外圆，次大视作内圆
    circles_sorted = sorted(circles.tolist(), key=lambda c: c[2], reverse=True)

    if len(circles_sorted) == 1:
        raise RuntimeError("build_ring_mask: 只检测到一个圆，建议调整 Hough 参数。")

    (xo, yo, ro) = circles_sorted[0]  # outer
    (xi, yi, ri) = circles_sorted[1]  # inner

    # 用两个圆的圆心取平均作为统一圆心（一般两者非常接近）
    cx = int(round((xo + xi) / 2))
    cy = int(round((yo + yi) / 2))
    r_outer = int(max(ro, ri))
    r_inner = int(min(ro, ri))

    # 外圆内先全部填 255，再抠掉内圆，得到环形
    cv2.circle(mask, (cx, cy), r_outer, 255, thickness=-1)
    cv2.circle(mask, (cx, cy), r_inner, 0, thickness=-1)

    return mask

def clean_defect_mask(
    defect_bin: np.ndarray,
    kernel_size: int = 5,
    open_iters: int = 1,
    close_iters: int = 1,
) -> np.ndarray:
    """
    对二值缺陷图做形态学开运算 + 闭运算，去小噪点。
    """
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    defect_clean = defect_bin.copy()
    if open_iters > 0:
        defect_clean = cv2.morphologyEx(defect_clean,cv2.MORPH_OPEN,
                                        kernel_small,iterations=open_iters,)
    if close_iters > 0:
        defect_clean = cv2.morphologyEx(defect_clean,cv2.MORPH_CLOSE,
                                        kernel_small,iterations=close_iters,)
    return defect_clean

def highlight_defect_edges(
    img,defect_clean,image_edge=None,min_area = 50,max_area = 5000,edge_kernel_size= 3):
    """
    对清洗后的二值缺陷图做连通域分析，筛掉过小/过大的区域，
    只对缺陷“边缘像素”上色，返回高亮图和边缘 mask。
    输入:
        img_bgr        : 原始img图
        defect_clean   : 清洗后的二值缺陷图 (0 / 255)
        image_edge     : 加边缘框的图像
        min_area, max_area: 面积筛选范围
        edge_kernel_size : 用于提取边缘的梯度核大小
    返回:
        vis        : 在原图基础上高亮了缺陷边缘的图 (BGR)
        edge_mask  : 仅缺陷边缘的二值 mask (0 / 255)
        defect_list: 缺陷信息列表，每个元素包含 index, bbox, center, area
    """
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        defect_clean, connectivity=8)

    mask_defects = np.zeros_like(defect_clean, dtype=np.uint8)
    defect_list: List[Dict[str, Any]] = []

    for i in range(1, num_labels):  # 从 1 开始跳过背景
        x, y, w, h, area = stats[i]

        if area < min_area or area > max_area:
            continue

        cx, cy = centroids[i]
        defect_list.append(
            {
                "index": i,
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (float(cx), float(cy)),
                "area": int(area),
            }
        )
        # 该连通域对应的像素置为 255
        mask_defects[labels == i] = 255
    # 形态学梯度提取“边缘像素”
    kernel_edge = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_kernel_size, edge_kernel_size))
    edge_mask = cv2.morphologyEx(
        mask_defects, cv2.MORPH_GRADIENT, kernel_edge)

    if image_edge is not None:
        vis = image_edge
    else:
        vis = img
    # 只对边缘像素上色
    vis[edge_mask == 255] = (0, 0, 255)  # BGR 红色

    # 在图像上标注面积信息
    for d in defect_list:
        cx, cy = d["center"]
        area = d["area"]

        text = f"S={str(area)}pixel"  # 也可以改成 f"A={area}"
        # 文本位置略微偏移一点，避免刚好压在黑点中心
        org = (int(cx) - 50, int(cy) - 10)

        cv2.putText(
            vis,text,org,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,  # 字体缩放
            (128, 0, 128),  # 紫色字体
            4,  # 线宽
            cv2.LINE_AA,
        )

    return vis , edge_mask, defect_list

def detect_defects(
    image_path: str,
    image_edge = None,
    bottom_hat_kernel_size: int = 35,
    k_std: float = 4.0,
    clean_kernel_size: int = 5,
    open_iters: int = 1,
    close_iters: int = 1,
    min_area: int = 50,
    max_area: int = 5000,
    edge_kernel_size: int = 2,
    save_path: Optional[str] = None,
):
    """
    输入:
        image_path : 输入图片路径
        image_edge : 已绘制边缘圆环的图像
        bottom_hat_kernel_size: 底帽变换的结构元素大小（越大越能突出小暗斑）
        k_std      : 阈值 = 均值 + k_std * 标准差
        clean_kernel_size : 形态学清洗的核大小
        open_iters, close_iters: 开运算/闭运算迭代次数
        min_area, max_area     : 缺陷面积筛选范围
        edge_kernel_size       : 提取缺陷边缘用的梯度核大小
        save_path              : 若不为 None，则把高亮结果保存到该路径
    返回:
        一个 dict，包含：
        img_bgr
        img_gray
        mask_ring
        ring_gray
        bottom_hat
        defect_bin
        defect_clean
        edge_mask
        vis (高亮结果)
        defect_list (缺陷信息)
    """
    #先用 edge_detection 模块做内外圆检测
    img_bgr, img_gray, edges, circles = run_edge_detection(image_path)
    if circles is None or len(circles) < 2:
        raise RuntimeError("detect_defects: 未能检测到内外两个圆，请先检查 edge_detection 参数。")

    #构造环形掩膜 & 抠出环形区域灰度
    mask_ring = build_ring_mask(img_gray, circles)
    ring_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask_ring)

    #底帽变换，突出偏暗小缺陷
    kernel_bh = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (bottom_hat_kernel_size, bottom_hat_kernel_size),
    )
    closing = cv2.morphologyEx(ring_gray, cv2.MORPH_CLOSE, kernel_bh)
    bottom_hat = cv2.subtract(closing, ring_gray)

    #在环形区域内做简单统计阈值
    mask_bool = mask_ring > 0
    values = bottom_hat[mask_bool]

    mean_val = float(values.mean())
    std_val = float(values.std())
    thresh_val = mean_val + k_std * std_val

    _, defect_bin = cv2.threshold(
        bottom_hat, thresh_val, 255, cv2.THRESH_BINARY
    )
    defect_bin = cv2.bitwise_and(defect_bin, defect_bin, mask=mask_ring)

    #形态学开闭操作，得到干净的二值缺陷图
    defect_clean = clean_defect_mask(
        defect_bin,
        kernel_size=clean_kernel_size,
        open_iters=open_iters,
        close_iters=close_iters,
    )

    #连通域 + 边缘高亮
    vis, edge_mask, defect_list = highlight_defect_edges(
        img_bgr,
        defect_clean,
        image_edge = image_edge,
        min_area=min_area,
        max_area=max_area,
        edge_kernel_size=edge_kernel_size,
    )

    print(f"通过面积筛选的缺陷数量: {len(defect_list)}")

    # 保存高亮结果
    if save_path is not None:
        ok = cv2.imwrite(save_path, vis)
        print(f"结果已保存到 {save_path}, 写入状态: {ok}")

    return {
        "img_bgr": img_bgr,
        "img_gray": img_gray,
        "mask_ring": mask_ring,
        "ring_gray": ring_gray,
        "bottom_hat": bottom_hat,
        "defect_bin": defect_bin,
        "defect_clean": defect_clean,
        "edge_mask": edge_mask,
        "vis": vis,
        "defect_list": defect_list,
    }


if __name__ == "__main__":
    test_image_path = r"./stainimg/cam1_20230302132318531.bmp"

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"测试图片不存在，请修改路径: {test_image_path}")

    # 输出结果路径（可选）
    dir_name, file_name = os.path.split(test_image_path)
    name, ext = os.path.splitext(file_name)

    out_dir = "out"
    out_path = os.path.join(out_dir, f"{name}_defect_end_edges{ext}")

    results = detect_defects(
        test_image_path,
        bottom_hat_kernel_size=35,
        k_std=4.0,
        clean_kernel_size=5,
        open_iters=1,
        close_iters=1,
        min_area=50,
        max_area=5000,
        edge_kernel_size=3,
        save_path=out_path,
    )

    img_gray = results["img_gray"]
    mask_ring = results["mask_ring"]
    ring_gray = results["ring_gray"]
    bottom_hat = results["bottom_hat"]
    defect_clean = results["defect_clean"]
    edge_mask = results["edge_mask"]
    vis = results["vis"]

    # 可视化
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Gray Image")
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Ring Mask")
    plt.imshow(mask_ring, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Ring Gray")
    plt.imshow(ring_gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Bottom-hat")
    plt.imshow(bottom_hat, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Defect Clean (Binary)")
    plt.imshow(defect_clean, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Defect Edges Highlighted")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
