import os
from typing import List, Dict, Any, Tuple
import cv2
from ed_detect import run_edge_detection
from df_detect import detect_defects

def process_folder(input_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # 只处理 .bmp 文件
    image_names = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(".bmp")
    ]
    total_images = 0
    total_defects = 0
    per_image_stats: List[Tuple[str, int, List[Dict[str, Any]]]] = []

    for img_name in image_names:
        img_path = os.path.join(input_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        print(f"\n==== 处理图像: {img_name} ====")
        total_images += 1
        name, ext = os.path.splitext(img_name)
        #边框检测（内外圆）
        try:
            img_bgr, img_gray, edges, circles = run_edge_detection(img_path)
        except Exception as e:
            print(f"[ERROR] run_edge_detection 失败: {e}")
            # 这张图直接跳过缺陷检测，继续下一张
            per_image_stats.append((img_name, 0, []))
            continue

        # 在原图上画出检测到的圆（第一个红色，第二个绿色）
        circle_vis = img_bgr.copy()
        if circles is not None and len(circles) > 0:
            colors = [(0, 0, 255), (0, 255, 0)]  # BGR: 红 / 绿
            for idx, (x, y, r) in enumerate(circles):
                color = colors[idx] if idx < len(colors) else (255, 0, 0)
                cv2.circle(circle_vis, (int(x), int(y)), int(r), color, 10)
                cv2.circle(circle_vis, (int(x), int(y)), 20, color, -1)

        # 缺陷检测
        defects_out_path = os.path.join(out_dir, f"{name}_defects{ext}")
        try:
            results = detect_defects(
                img_path,
                image_edge = circle_vis,
                bottom_hat_kernel_size=35,
                k_std=4.0,
                clean_kernel_size=5,
                open_iters=1,
                close_iters=1,
                min_area=50,
                max_area=5000,
                edge_kernel_size=3,
                save_path=defects_out_path,
            )
        except Exception as e:
            print(f"[ERROR] detect_defects 失败: {e}")
            per_image_stats.append((img_name, 0, []))
            continue

        defect_list = results.get("defect_list", [])
        num_defects = len(defect_list)
        total_defects += num_defects

        per_image_stats.append((img_name, num_defects, defect_list))

    #生成检测报告
    report_path = os.path.join(out_dir, "detection_report.txt")
    write_report(report_path, total_images, total_defects, per_image_stats)
    print(f"\n检测完成，报告已保存到: {report_path}")


def write_report(report_path,total_images,total_defects,per_image_stats,):
    """
    生成 txt 检测报告：
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==== 缺陷检测报告 ====\n\n")
        f.write(f"检测图片总数量: {total_images}\n")
        f.write(f"检测到的缺陷总数: {total_defects}\n\n")

        for img_name, num_defects, defect_list in per_image_stats:
            f.write(f"图像文件: {img_name}\n")
            f.write(f"  缺陷数量: {num_defects}\n")

            for idx, d in enumerate(defect_list, start=1):
                area = d.get("area", -1)
                center = d.get("center", (-1.0, -1.0))
                bbox = d.get("bbox", (-1, -1, -1, -1))
                cx, cy = center
                x, y, w, h = bbox
                f.write(f"    缺陷 {idx}:\n")
                f.write(f"      面积: {area} 像素\n")
                f.write(f"      中心位置: (x={cx:.1f}, y={cy:.1f})\n")
                f.write(
                    f"      外接矩形: "
                    f"(x={x}, y={y}, w={w}, h={h})\n"
                )
            f.write("\n")


def main():
    input_dir = r"./stainimg"   # 放待检测 bmp 图片的文件夹
    out_dir = r"./out"          # 输出结果的文件夹

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    process_folder(input_dir, out_dir)


if __name__ == "__main__":
    main()
