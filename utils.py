# -*- encoding: utf-8 -*-
'''@File    :   utils.py
@Author  :   编程学习园地 
@License :   该项目受专利、软著保护，仅供个人学习使用，严禁倒卖，一经发现，编程学习园地团队有必要追究法律责任！！！
'''

#兼容不同版本的 ultralytics：新版已把 YOLOv10 并入 YOLO 类，旧版单独导出 YOLOv10
try:
    from ultralytics import YOLOv10
except ImportError:
    from ultralytics import YOLO as YOLOv10
import streamlit as st
import numpy as np
from PIL import Image
import cv2

@st.cache_resource
def load_model(model_path):
    """
    从具体的路径中加载一个模型
    """
    model=YOLOv10(model_path)
    return model
 
def infer_image(model, image, conf, iou, is_save=False):
    '''
    图片推理 - 采用最严格的验证方法，确保传递给模型的绝对是有效的numpy数组
    '''
    try:
        # 1. 输入类型检查（静默模式）
        
        # 2. 创建一个绝对有效的默认图像（作为最后防线）
        def get_absolutely_safe_image():
            # 直接创建一个完美的numpy数组
            safe_img = np.ones((640, 640, 3), dtype=np.uint8) * 255  # 白色RGB图像
            # 强制确保连续性
            safe_img = np.ascontiguousarray(safe_img)
            # 再次验证
            if not isinstance(safe_img, np.ndarray) or not safe_img.flags.contiguous:
                # 如果还不行，尝试另一种方式创建
                safe_img = np.zeros((640, 640, 3), dtype=np.uint8)
                safe_img = np.ascontiguousarray(safe_img)
            return safe_img
        
        # 3. 从PIL图像直接转换为numpy数组的安全方法
        try:
            # 确保我们处理的是PIL图像
            if hasattr(image, 'convert'):
                # 确保图像是RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # 转换为numpy数组
                img_array = np.array(image)
            else:
                # 如果不是PIL图像，尝试直接转换
                img_array = np.array(image)
            
            # 4. 立即进行严格验证
            if not isinstance(img_array, np.ndarray):
                st.error("严重错误：转换结果不是numpy数组")
                img_array = get_absolutely_safe_image()
            elif img_array.size == 0:
                st.error("严重错误：numpy数组为空")
                img_array = get_absolutely_safe_image()
        except Exception as e:
            st.error(f"转换失败: {str(e)}")
            # 发生任何错误，立即使用最安全的默认图像
            img_array = get_absolutely_safe_image()
        
        # 5. 确保数据类型
        try:
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
        except:
            st.error("数据类型转换失败")
            img_array = get_absolutely_safe_image()
        
        # 6. 强制确保连续性 - 这是关键步骤
        img_array = np.ascontiguousarray(img_array)
        
        # 7. 处理通道数 - 使用最简单的方法
        try:
            if len(img_array.shape) == 2:
                # 灰度图转RGB
                img_array = np.dstack([img_array, img_array, img_array])
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    # RGBA转RGB
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] != 3:
                    # 不支持的通道数
                    st.error(f"不支持的通道数: {img_array.shape[2]}")
                    img_array = get_absolutely_safe_image()
            else:
                # 错误的维度
                st.error(f"错误的图像维度: {img_array.shape}")
                img_array = get_absolutely_safe_image()
        except:
            st.error("通道处理失败")
            img_array = get_absolutely_safe_image()
        
        # 8. 最终的最终验证 - 确保万无一失
        if not isinstance(img_array, np.ndarray):
            st.error("致命错误：图像不是numpy数组")
            img_array = get_absolutely_safe_image()
        
        # 再次强制连续性
        img_array = np.ascontiguousarray(img_array)
        
        # 9. 最终图像验证（静默模式）
        
        # 使用numpy数组进行推理
        res = model.predict(source=img_array, conf=conf, iou=iou, save=is_save)
        
        # 获取注释后的图像（这一步会返回numpy数组）
        anno_img = res[0].plot()
        
        # 提取结果信息
        labels = res[0].names
        boxes = res[0].boxes
        labels_num_dict = {}
        rows = []
        
        # 处理检测框信息
        for index, box in enumerate(boxes):
            row = []
            lable_index = box.cls.cpu().detach().numpy()[0].astype(int)
            label_name = labels[lable_index]
            xyxy = box.xyxy.cpu().tolist()[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            confidence = box.conf.cpu().detach().numpy()[0]
            conf = "{:.2f}".format(confidence)
            row.append(index + 1)
            row.append(label_name)
            row.append(conf)
            row.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            rows.append(row)
            
            # 更新类别计数
            if label_name in labels_num_dict:
                labels_num_dict[label_name] += 1
            else:
                labels_num_dict[label_name] = 1
                
        return anno_img, labels_num_dict, rows
        
    except Exception as e:
        # 捕获并记录错误
        st.error(f"图像推理出错: {str(e)}")
        # 返回默认红色图像以显示错误
        error_img = Image.new('RGB', (640, 640), color='red')
        return np.array(error_img), {}, []
            
def infer_video_frame(model,image,conf,iou,is_save=False):
    '''
    视频推理和本地摄像头推理
    '''
    try:
        # 创建默认图像函数
        def create_default_image():
            return np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 安全检查：在任何处理前先确保我们有一个有效的图像
        safe_frame = None
        
        # 检查是否为PIL Image对象
        if isinstance(image, Image.Image):
            # 转换PIL Image到numpy数组
            try:
                safe_frame = np.array(image)
                # 检查转换后的数组
                if safe_frame is None or not isinstance(safe_frame, np.ndarray) or safe_frame.size == 0:
                    st.warning("PIL图像转换为numpy数组失败，使用默认图像")
                    safe_frame = create_default_image()
            except Exception as e:
                st.error(f"转换PIL图像时出错: {str(e)}，使用默认图像")
                safe_frame = create_default_image()
        # 检查是否已经是numpy数组
        elif isinstance(image, np.ndarray):
            safe_frame = image.copy()  # 创建副本以避免修改原始数据
        else:
            # 尝试转换为numpy数组
            try:
                safe_frame = np.array(image)
                if safe_frame is None or not isinstance(safe_frame, np.ndarray) or safe_frame.size == 0:
                    st.warning("无法转换为有效的numpy数组，使用默认图像")
                    safe_frame = create_default_image()
            except Exception as e:
                st.error(f"转换为numpy数组时出错: {str(e)}，使用默认图像")
                safe_frame = create_default_image()
        
        # 确保我们有一个有效的numpy数组
        if not isinstance(safe_frame, np.ndarray):
            st.error("致命错误：无法创建有效的numpy数组，使用默认图像")
            safe_frame = create_default_image()
        
        # 确保数组不为空
        if safe_frame.size == 0:
            st.warning("视频帧数组为空，使用默认图像")
            safe_frame = create_default_image()
        
        # 确保图像有正确的维度
        if len(safe_frame.shape) not in [2, 3]:
            st.warning(f"视频帧维度异常: {safe_frame.shape}，使用默认图像")
            safe_frame = create_default_image()
        
        # 确保数据类型正确
        if safe_frame.dtype != np.uint8:
            try:
                safe_frame = safe_frame.astype(np.uint8)
                st.warning("视频帧数据类型已转换为uint8")
            except:
                st.error("无法转换视频帧数据类型，使用默认图像")
                safe_frame = create_default_image()
        
        # 确保数组是连续的
        if not safe_frame.flags.contiguous:
            safe_frame = np.ascontiguousarray(safe_frame)
        
        # 处理通道数
        if len(safe_frame.shape) == 2:
            # 灰度图转RGB
            try:
                safe_frame = cv2.cvtColor(safe_frame, cv2.COLOR_GRAY2RGB)
            except:
                st.warning("灰度图转换失败，使用默认RGB图像")
                safe_frame = create_default_image()
        elif len(safe_frame.shape) == 3:
            if safe_frame.shape[2] == 4:
                # RGBA转RGB
                try:
                    safe_frame = cv2.cvtColor(safe_frame, cv2.COLOR_RGBA2RGB)
                except:
                    st.warning("RGBA转RGB失败，使用默认RGB图像")
                    safe_frame = create_default_image()
            elif safe_frame.shape[2] != 3:
                # 其他通道数，使用默认图像
                st.warning(f"不支持的通道数: {safe_frame.shape[2]}，使用默认RGB图像")
                safe_frame = create_default_image()
        
        # 最终验证
        if not isinstance(safe_frame, np.ndarray) or safe_frame.size == 0 or not safe_frame.flags.contiguous:
            st.error("最终视频帧验证失败，使用默认图像")
            safe_frame = create_default_image()
        
        # 执行预测 - 使用经过验证的safe_frame
        res = model.predict(source=safe_frame, conf=conf, iou=iou)
        anno_img = res[0].plot()
        return anno_img
        
    except Exception as e:
        # 发生错误时返回默认图像
        st.error(f"视频帧处理出错: {str(e)}")
        return np.zeros((640, 640, 3), dtype=np.uint8)