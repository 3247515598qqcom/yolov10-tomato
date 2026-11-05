import streamlit as st
from PIL import Image
from utils import load_model,infer_image,infer_video_frame
from config import *
import tempfile
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="番茄叶片病虫害检测系统",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
    )
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown("<h1 style='text-align: center;'>基于YOLO10的番茄叶片病虫害检测系统</h1>", unsafe_allow_html=True)
st.sidebar.header("配置面板")
model_type = st.sidebar.selectbox(
    "选取模型",
    MODEL_LIST
)
model_path = Path(MODEL_DIR, str(model_type))
confidence = float(st.sidebar.slider(
"调整置信度", 10, 100, 25)) / 100
iou=float(st.sidebar.slider(
    "调整iou", 10, 100, 45)) / 100
source = ("图片检测", "视频检测",'本地摄像头检测')
select_radio=st.sidebar.radio('检测类型',source)
source_index=source.index(select_radio)
model=load_model(model_path)
#图像输入 → 预处理（尺寸调整/归一化）→ YOLO10模型推理 → 后处理（NMS过滤）→ 结果可视化
if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            with st.spinner(text='图片推理执行中...'):
                st.sidebar.image(uploaded_file,caption="原始图片")
                # 直接使用PIL图像对象，不转换为numpy数组
                try:
                    picture = Image.open(uploaded_file)
                    # 确保图像模式为RGB（处理可能的RGBA或灰度图像）
                    if picture.mode == 'RGBA':
                        # 创建一个白色背景
                        background = Image.new('RGB', picture.size, (255, 255, 255))
                        # 将RGBA图像粘贴到白色背景上，使用alpha通道作为掩码
                        background.paste(picture, mask=picture.split()[3])
                        picture = background
                    elif picture.mode == 'L':
                        # 灰度图转RGB
                        picture = picture.convert('RGB')
                    elif picture.mode != 'RGB':
                        # 其他模式转RGB
                        picture = picture.convert('RGB')
                    
                    # 移除图像信息显示
                    
                    # 直接传递PIL图像对象
                    anno_img, label_num_dict,rows = infer_image(model=model, image=picture, conf=confidence, iou=iou)
                except Exception as e:
                    st.error(f"图像加载或预处理出错: {str(e)}")
                    # 创建一个简单的错误提示图像
                    error_img = Image.new('RGB', (640, 640), color='red')
                    anno_img = np.array(error_img)
                    label_num_dict = {} 
                    rows = []
                st.image(anno_img,channels='BGR',caption="图片检测结果")
                df=pd.DataFrame(rows,columns=['序号','类别','置信度','box坐标'])
                df1=pd.DataFrame(list(label_num_dict.items()),columns=['类别','总数'])
                st.data_editor(df,hide_index=True)
                st.data_editor(df1)
                    
elif source_index==1:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            st.sidebar.video(uploaded_file)#视频上传与用户交互
            #写入系统临时目录，避免往项目目录写文件（云端部署时项目目录通常不可写）
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            vid_cap = cv2.VideoCapture(tfile.name)#创建临时文件并写入视频帧
            st_frame = st.empty()
            with st.spinner(text='视频推理执行中...'):
                frame_rate_divider = 2  # 设置帧率除数
                frame_count = 0  # 初始化帧计数器
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if not success:
                        vid_cap.release()
                        break
                    # 调用方法逐帧预测
                    if frame_count % frame_rate_divider == 0:
                        anno_img=infer_video_frame(model=model,image=image,conf=confidence,iou=iou,is_save=False)
                        st_frame.image(anno_img, channels='BGR', caption="视频检测结果")
                    frame_count += 1
else:
        with st.spinner(text='摄像头推理执行中...'):
            flag = st.button(
                label="终止执行"
            )
            while not flag:
                vid_cap = cv2.VideoCapture(0)  # 调用本地摄像头
                # 页面创建空容器实时播放画面
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        # 调用方法逐帧预测
                        anno_img=infer_video_frame(model=model, image=image, conf=confidence, iou=iou,is_save=False)
                        st_frame.image(anno_img, channels='BGR', caption="摄像头检测结果")
                    else:
                        vid_cap.release()
                        break