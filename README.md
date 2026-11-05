# 基于 YOLOv10 的番茄叶片病虫害检测系统

基于 YOLOv10 目标检测模型与 Streamlit 构建的番茄叶片病虫害识别系统，支持**图片检测、视频检测、摄像头实时检测**三种模式，输出检测框、类别、置信度与类别统计。

适用于田间快速病害初筛场景，单张图片推理约 0.15s，视频检测帧率 25–30 FPS。

## 检测效果

示例输入图片与演示视频见 `img_video/` 目录，可直接上传用于功能验证。

## 功能特性

- **三种检测模式**：图片检测 / 视频逐帧检测 / 本地摄像头实时检测
- **参数可调**：侧边栏滑动条调节置信度阈值（conf）与 IoU 阈值，实时生效
- **结果结构化**：检测结果以表格输出，包含序号、类别、置信度、检测框坐标，并统计各类别数量
- **多模型切换**：将自训练权重放入 `weights/` 目录，侧边栏下拉框自动识别并可选（支持 `.pt` 与 `.onnx`）
- **健壮的图像预处理**：兼容 RGB / RGBA / 灰度图像，统一转 RGB 并保证数组连续，避免推理阶段类型异常

## 技术栈

| 类别 | 技术 |
|------|------|
| 目标检测 | YOLOv10（Ultralytics）、PyTorch |
| 图像处理 | OpenCV、Pillow、NumPy |
| Web 交互 | Streamlit |
| 数据处理 | Pandas |

## 项目结构

```
yolov10-tomato/
├─ app.py             # Streamlit 主程序：页面布局与三种检测模式
├─ config.py          # 路径配置、设备选择（GPU/CPU）、模型列表加载
├─ utils.py           # 模型加载、图片推理、视频/摄像头逐帧推理封装
├─ weights/
│  └─ best.pt         # 训练好的 YOLOv10 权重（5.6 MB）
├─ img_video/         # 演示视频与示例图片
├─ requirements.txt
└─ README.md
```

## 快速开始

```bash
# 1. 安装依赖（建议使用 Python 3.9 - 3.11）
pip install -r requirements.txt

# 2. 启动
streamlit run app.py
```

浏览器打开终端提示的地址（默认 http://localhost:8501），在侧边栏调整参数后选择检测类型即可。

若使用 GPU 推理，请自行安装与 CUDA 版本匹配的 PyTorch；无 GPU 时自动降级为 CPU。

## 在线部署

**Streamlit Community Cloud（推荐，免费）**

1. 将本仓库推送到 GitHub
2. 打开 <https://share.streamlit.io>，使用 GitHub 账号登录
3. New app → 选择仓库 → Main file path 填 `app.py` → Deploy
4. 等待依赖安装完成，即可获得公网访问链接

**Hugging Face Spaces**

新建 Space，SDK 选择 Streamlit，上传本目录全部文件（`best.pt` 一并上传）即可运行。

## 数据集说明

训练数据为番茄叶片病虫害图像，共 **9 个类别**、约 **6700 张**图片，按 train / valid / test 划分，总量约 212 MB。

```yaml
nc: 9
names: ['Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold',
        'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus']
```

数据集体积较大，未包含在本仓库中（已在 `.gitignore` 中排除）。如需复现训练，请自备数据并按上述类别组织为 YOLO 格式目录，`data.yaml` 中 `train` / `val` 指向对应的 `images` 目录。

## 说明

- 本项目为个人学习与实践项目，检测结果仅供辅助参考，不作为农业诊断的最终依据。
- 界面框架基于公开的 Streamlit 检测模板二次开发，已完成路径适配与云端部署改造。
