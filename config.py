import os
import sys
from pathlib import Path
import torch

#获取当前文件的绝对路径
file_path = Path(__file__).resolve()

#获取当前文件的上一级目录的路径
root_path = file_path.parent

#如果当前文件的父目录不在搜索路径中则添加进去
if root_path not in sys.path:
    sys.path.append(str(root_path))

#项目根目录：直接取当前文件所在目录的绝对路径，不依赖运行时的工作目录
ROOT = root_path
MODEL_DIR = ROOT / 'weights'
#注意：如果你想要加载自己训练的模型(yolov5、v8微调版本(必须是通过ultralytics训练),这些都支持,修改了网络结构的不支持;且要注意模型训练版本与推理版本所使用的ultralytics相同)
# 只需将自己的模型放在weights目录下,系统界面'模型选择'下拉框会显示出新模型，届时可以手动选择模型

#侧边栏模型选择列表
MODEL_LIST=[]
if torch.cuda.is_available():
    device='gpu'
else:
    device='cpu'
weights=os.listdir(MODEL_DIR)
#支持pytroch原生推理与onnx推理,若为gpu环境,pt模型放在前面默认加载pt模型执行推理,除非手动选择模型;
#若为cpu环境,onnx模型放在前面默认加载onnx模型执行推理,除非手动选择模型
if device=='cpu':
    for weight in weights:
        if weight.endswith('.onnx'):
            MODEL_LIST.append(weight)
    for weight in weights:
        if weight.endswith('.pt'):
            MODEL_LIST.append(weight)
else:
    for weight in weights:
        if weight.endswith('.pt'):
            MODEL_LIST.append(weight)
    for weight in weights:
        if weight.endswith('.onnx'):
            MODEL_LIST.append(weight)