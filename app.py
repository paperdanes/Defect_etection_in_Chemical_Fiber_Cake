import os
import io
import tempfile
import zipfile
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from df_detect import detect_defects
from ed_detect import run_edge_detection

# ============ Streamlit 页面配置 ============ #
st.set_page_config(
    page_title="工业缺陷检测可视化平台",
    layout="wide",
)


def generate_single_image_report(
    image_name: str,
    defect_list: List[Dict[str, Any]],
) -> str:
    """
    根据单张图片的缺陷列表，生成文本形式的检测报告。
    """
    buf = io.StringIO()
    buf.write("==== 单张图片缺陷检测报告 ====\n")
    buf.write(f"图像文件: {image_name}\n")
    buf.write(f"缺陷数量: {len(defect_list)}\n\n")

    for idx, d in enumerate(defect_list, start=1):
        area = d.get("area", -1)
        center = d.get("center", (-1.0, -1.0))
        bbox = d.get("bbox", (-1, -1, -1, -1))
        cx, cy = center
        x, y, w, h = bbox

        buf.write(f"  缺陷 {idx}:\n")
        buf.write(f"    面积: {area} 像素\n")
        buf.write(f"    中心位置: (x={cx:.1f}, y={cy:.1f})\n")
        buf.write(f"    外接矩形: (x={x}, y={y}, w={w}, h={h})\n\n")

    return buf.getvalue()


def show_single_image_page():
    st.header("单张图片缺陷检测")

    uploaded_file = st.file_uploader(
        "上传待检测图像（支持 bmp / png / jpg / jpeg / tif）",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("请在上方上传一张图片。")
        return

    # 在页面上先显示原始图像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("无法解析该图像，请确认文件格式是否正确。")
        return

    st.subheader("原始图像预览")
    st.image(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        caption="原始图像",
        use_container_width=True,
    )

    # 将上传的文件写入临时路径，供现有检测函数使用
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    if st.button("开始检测", type="primary"):
        try:
            # 先做一次边缘检测 + 圆检测，并把圆画在图上
            img_bgr_edge, img_gray, edges, circles = run_edge_detection(tmp_path)

            circle_vis = img_bgr_edge.copy()
            if circles is not None and len(circles) > 0:
                colors = [(0, 0, 255), (0, 255, 0)]  # BGR: 红 / 绿
                for idx, (x, y, r) in enumerate(circles):
                    color = colors[idx] if idx < len(colors) else (255, 0, 0)
                    cv2.circle(circle_vis, (int(x), int(y)), int(r), color, 10)
                    cv2.circle(circle_vis, (int(x), int(y)), 20, color, -1)

            # 画好圆的 circle_vis 作为 image_edge 传给缺陷检测
            results = detect_defects(
                tmp_path,
                image_edge=circle_vis,
                bottom_hat_kernel_size=35,
                k_std=4.0,
                clean_kernel_size=5,
                open_iters=1,
                close_iters=1,
                min_area=50,
                max_area=5000,
                edge_kernel_size=3,
                save_path=None,
            )
        except Exception as e:
            st.error(f"检测过程中出现错误：{e}")
            return
        finally:
            # 临时文件用完后删除
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        defect_list = results.get("defect_list", [])
        vis = results.get("vis", None)

        st.subheader("缺陷检测结果图")
        if vis is not None:
            st.image(
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                caption="缺陷高亮结果（已叠加圆）",
                use_container_width=True,
            )
        else:
            st.warning("未获得结果图像。")

        # 结果统计与表格
        st.subheader("检测统计信息")
        st.write(f"检测到的缺陷数量：**{len(defect_list)}**")

        if defect_list:
            df_rows = []
            for d in defect_list:
                area = d.get("area", -1)
                center = d.get("center", (-1.0, -1.0))
                bbox = d.get("bbox", (-1, -1, -1, -1))
                cx, cy = center
                x, y, w, h = bbox

                df_rows.append(
                    {
                        "面积（像素）": area,
                        "中心 X": cx,
                        "中心 Y": cy,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_w": w,
                        "bbox_h": h,
                    }
                )

            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True)

            # 生成文本报告（不展示，只提供下载）
            report_text = generate_single_image_report(
                uploaded_file.name, defect_list
            )

            st.download_button(
                label="下载检测报告（TXT）",
                data=report_text.encode("utf-8"),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_report.txt",
                mime="text/plain",
            )

            # 检测结果图片下载按钮
            if vis is not None:
                # 用 PNG 编码到内存
                success, img_buf = cv2.imencode(".bmp", vis)
                if success:
                    st.download_button(
                        label="下载检测结果图片（PNG）",
                        data=img_buf.tobytes(),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_defect_result.png",
                        mime="image/bmp",
                    )
        else:
            st.info("未检测到符合条件的缺陷。")



def show_batch_zip_page():
    st.header("批量图片检测（压缩包）")
    st.markdown(
        """
        **使用说明：**
        - 上传一个 `.zip` 压缩包；
        - 压缩包内放置待检测的 **bmp / png / jpg / jpeg / tif** 图像；
        - 系统将自动完成所有图片的检测，并生成整体检测报告。
        """
    )

    uploaded_zip = st.file_uploader(
        "上传包含多张图片的压缩包（.zip）",
        type=["zip"],
        accept_multiple_files=False,
    )

    if uploaded_zip is None:
        st.info("请在上方上传一个 zip 压缩包。")
        return

    if st.button("开始批量检测", type="primary"):
        # 为文件解压和处理创建临时目录
        base_dir = tempfile.mkdtemp(prefix="defect_batch_")
        input_dir = os.path.join(base_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # 将 zip 内容解压到 input_dir
        try:
            with zipfile.ZipFile(uploaded_zip) as zf:
                zf.extractall(input_dir)
        except Exception as e:
            st.error(f"解压 zip 文件失败：{e}")
            return

        # 收集所有待检测图片路径
        exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        image_paths = []
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(root, fname))

        if not image_paths:
            st.error("压缩包中未找到可识别的图片文件（bmp/png/jpg/jpeg/tif）。")
            return

        image_paths = sorted(image_paths)
        total_images = len(image_paths)

        progress_bar = st.progress(0)
        status_text = st.empty()

        all_rows = []         # 用于 DataFrame，每一行一个缺陷
        preview_images = []   # 存放少量预览结果
        report_buf = io.StringIO()

        for img_idx, img_path in enumerate(image_paths, start=1):
            # img_idx 就是“第几张图片”
            img_name = os.path.relpath(img_path, input_dir)
            status_text.text(f"正在检测 第 {img_idx}/{total_images} 张: {img_name}")

            try:
                # 批量检测里，同样先做边缘检测 + 画圆
                img_bgr_edge, img_gray, edges, circles = run_edge_detection(img_path)
                circle_vis = img_bgr_edge.copy()
                if circles is not None and len(circles) > 0:
                    colors = [(0, 0, 255), (0, 255, 0)]
                    for c_idx, (x, y, r) in enumerate(circles):
                        color = colors[c_idx] if c_idx < len(colors) else (255, 0, 0)
                        cv2.circle(circle_vis, (int(x), int(y)), int(r), color, 10)
                        cv2.circle(circle_vis, (int(x), int(y)), 20, color, -1)

                # 将 circle_vis 传入 detect_defects，得到“圆 + 缺陷”叠加结果
                results = detect_defects(
                    img_path,
                    image_edge=circle_vis,
                    bottom_hat_kernel_size=35,
                    k_std=4.0,
                    clean_kernel_size=5,
                    open_iters=1,
                    close_iters=1,
                    min_area=50,
                    max_area=5000,
                    edge_kernel_size=3,
                    save_path=None,
                )
            except Exception as e:
                # 记录错误信息（仍然占一行）
                all_rows.append(
                    {
                        "图片序号": img_idx,
                        "图片名": img_name,
                        "缺陷序号": None,
                        "面积（像素）": None,
                        "中心 X": None,
                        "中心 Y": None,
                        "bbox_x": None,
                        "bbox_y": None,
                        "bbox_w": None,
                        "bbox_h": None,
                        "备注": f"检测失败: {e}",
                    }
                )
                progress_bar.progress((img_idx) / total_images)
                continue

            defect_list = results.get("defect_list", [])
            vis = results.get("vis", None)

            # 表格行：每个缺陷一行；若没有缺陷，也记录一行
            if defect_list:
                for defect_idx, d in enumerate(defect_list, start=1):
                    area = d.get("area", -1)
                    center = d.get("center", (-1.0, -1.0))
                    bbox = d.get("bbox", (-1, -1, -1, -1))
                    cx, cy = center
                    x, y, w, h = bbox

                    all_rows.append(
                        {
                            "图片序号": img_idx,          # 第几张图片
                            "图片名": img_name,
                            "缺陷序号": defect_idx,       # 该图的第几个缺陷
                            "面积（像素）": area,
                            "中心 X": cx,
                            "中心 Y": cy,
                            "bbox_x": x,
                            "bbox_y": y,
                            "bbox_w": w,
                            "bbox_h": h,
                            "备注": "",
                        }
                    )
            else:
                all_rows.append(
                    {
                        "图片序号": img_idx,
                        "图片名": img_name,
                        "缺陷序号": 0,               # 0 表示该图没有缺陷
                        "面积（像素）": 0,
                        "中心 X": None,
                        "中心 Y": None,
                        "bbox_x": None,
                        "bbox_y": None,
                        "bbox_w": None,
                        "bbox_h": None,
                        "备注": "未检测到缺陷",
                    }
                )

            # 报告文本：复用单图报告生成函数
            report_buf.write(generate_single_image_report(img_name, defect_list))
            report_buf.write("\n")

            # 收集少量预览图（只取前 6 张）
            if vis is not None and len(preview_images) < 6:
                preview_images.append((img_name, vis.copy()))

            progress_bar.progress((img_idx) / total_images)

        status_text.text("批量检测完成。")

        # 生成总报告文本（不展示，只提供下载）
        batch_report_text = report_buf.getvalue()

        st.subheader("批量检测结果表")

        # 转 DataFrame，并按「图片序号 -> 缺陷序号」排序
        df_all = pd.DataFrame(all_rows)
        df_all = df_all.sort_values(
            by=["图片序号", "缺陷序号"],
            ascending=[True, True],
        ).reset_index(drop=True)

        st.dataframe(df_all, use_container_width=True)

        st.download_button(
            label="下载批量检测报告（TXT）",
            data=batch_report_text.encode("utf-8"),
            file_name="batch_detection_report.txt",
            mime="text/plain",
        )

        # 展示部分结果图
        st.subheader("部分结果图预览")
        if not preview_images:
            st.info("没有可展示的结果图。")
        else:
            cols = st.columns(3)
            for idx, (img_name, vis_img) in enumerate(preview_images):
                with cols[idx % 3]:
                    st.image(
                        cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB),
                        caption=img_name,
                        use_container_width=True,
                    )


def main():
    st.sidebar.title("工业缺陷检测可视化界面")
    mode = st.sidebar.radio(
        "选择功能",
        ("单张图片检测", "批量图片检测（压缩包）"),
    )

    if mode == "单张图片检测":
        show_single_image_page()
    else:
        show_batch_zip_page()


if __name__ == "__main__":
    main()
