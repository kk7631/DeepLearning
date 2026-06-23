## 数据集来源
本项目使用 PlantVillage 公开植物病害图像数据集中的玉米叶片相关类别。
数据集链接：<https://github.com/spMohanty/PlantVillage-Dataset>

## 文件说明
- `DL课程设计.ipynb`：课程设计报告主体。
- `corn_leaf_dataset/`：本项目使用的玉米叶片病害数据集，已按训练集、验证集和测试集划分。但数据集过大，故没有上传。
- `corn_predict_images/`：系统功能展示用图片，用于测试单张图片预测效果。
- `src/`：项目核心代码，包括数据读取、模型定义、训练、评估、预测和可视化分析等模块。
- `models/`：保存训练好的模型权重。
- `outputs/`：保存实验结果，包括训练曲线、分类报告、混淆矩阵、特征图、Grad-CAM 热力图和错误样本分析图等。
- `prepare_corn_leaf_dataset.py`：数据集下载与整理。
- `run_experiment.py`：模型训练、评估和主要结果生成。
- `generate_visualizations.py`：可视化分析结果生成。
