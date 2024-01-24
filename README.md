## 指南：数据处理、模型训练与预测

1. **数据集获取**：
   - 初始步骤包括下载指定数据集。本实例使用的是花卉分类数据集，可通过TensorFlow官方链接进行下载：[花卉数据集下载](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)。若遇到下载障碍，可使用百度云备用链接：[备用下载](https://pan.baidu.com/s/1QLCTA4sXnQAw_yvxPj9szg)，提取码：58p0。

2. **数据存储结构的构建**：
   - 在名为`data_set`的主文件夹中，创建一个子文件夹`flower_data`用于存放后续下载的数据集。
   - 将下载的数据集解压至`flower_data`文件夹。这将创建一个包含所有数据样本的文件夹，命名为`flower_photos`。
   - 执行`split_data.py`脚本，自动将数据集分割成训练集（train）和验证集（val）。

3. **训练脚本配置**：
   - 在`train.py`文件中，指定`--data-path`参数为解压后的`flower_photos`目录的绝对路径。

4. **预训练权重下载**：
   - 在`model.py`文件中，根据所选模型下载相应的预训练权重。

5. **权重路径设置**：
   - 在`train.py`脚本中，将`--weights`参数设置为下载的预训练权重文件路径。

6. **训练准备**：
   - 确保已设置数据集路径(`--data-path`)和预训练权重路径(`--weights`)后，启动`train.py`脚本进行模型训练。训练过程中将自动生成`class_indices.json`文件。

7. **预测脚本配置**：
   - 在`predict.py`脚本中，导入与训练脚本相同的模型，并设置`model_weight_path`为训练完成的模型权重路径（默认存于weights文件夹）。

8. **图片路径设置**：
   - 在`predict.py`脚本中，设置`img_path`为待预测图片的绝对路径。

9. **预测准备**：
   - 完成权重路径(`model_weight_path`)和预测图片路径(`img_path`)的设置后，运行`predict.py`脚本进行图像预测。

10. **自定义数据集使用**：
    - 若要使用个人数据集，请参考花分类数据集的文件结构（即每个类别对应一个文件夹）。调整训练和预测脚本中的`num_classes`参数，以匹配个人数据的类别数。

```
数据目录结构示例：
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，包含3670个样本）  
       ├── train（生成的训练集文件夹，包含3306个样本）  
       └── val（生成的验证集文件夹，包含364个样本）
```
