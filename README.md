# 工业缺陷检测作业（Industrial Defect Detection）

基于 **OpenCV + Streamlit** 的工业表面缺陷检测小项目，提供单视频/批量处理流程，并包含若干实验 Notebook 供算法对比与验证。<br>

#### 该代码为中北大学《机器视觉与图像处理》课设大作业，仅供参考和学习，祝大家都学有所成！

## 目录结构

```
Defect_etection_in_Chemical_Fiber_Cake_master
├─ app.py                  # Streamlit 可视化入口
├─ df_detect.py            # 缺陷检测（示例：污渍/斑点等）核心逻辑
├─ ed_detect.py            # 边缘/轮廓相关检测逻辑
├─ folder_process.py       # 文件夹批处理（视频/图片序列）
├─ out/                    # 输出目录（结果视频/图片/日志等）
├─ 缺陷检测.ipynb
└─ 边缘检测.ipynb
```
---

## 环境与依赖

- Python 3.9+（建议 3.10/3.11）
- 主要依赖：`opencv-python`、`streamlit`、`numpy`

---

## 快速开始

### 1) 安装依赖

```bash
conda create -n cv python==3.9
conda activate cv
pip install -r requirements.txt
```

### 2) 启动 Web 界面（Streamlit）

```bash
streamlit run app.py
```

---

## 输出说明

- 结果默认写入 `out/`：可包含标注后视频/帧、检测统计、过程日志等。
- 视频演示：【Bilibili】[基于Streamlit和OpenCV的化纤丝饼污渍类缺陷检测](https://www.bilibili.com/video/BV1HF2sBpEoN/)