# 多模态情感分析实验

## 环境配置
需要安装以下库：<br>
* pandas==1.4.4
* numpy==1.23.3
* tqdm
* glob2
* scikit-learn
* torch==2.0.0
* torchvision==0.15.1
* transformers==4.30.2  <br>
可直接在终端进入文件夹运行pip install -r requirements.txt<br>

## 文件结构
|-- data-->#包括所有的训练文本和图片，每个文件按照唯一的guid命名<br>
|-- README.md<br>
|-- requirements.txt<br>
|-- test_without_label.txt-->#数据的guid和对应的情感标签<br>
|-- train.txt-->#数据的guid和空的情感标签<br>
|-- 多模态情感分析.py-->#主函数<br>


## 运行方法
在终端中进入文件夹并输入：<br>
python 多模态情感分析.py --type_ xxx<br>
例：python 多模态情感分析.py --type_ 1 <br>
运行文件。<br>
注：type_决定训练模型的输入情况，1代表既输入文本也输入图像，2代表仅输入图像，3代表仅输入文本，不另输入则默认type_为1。<br>

## 参考
1.ResNet50模型 https://blog.csdn.net/Jackydyy/article/details/119238657 <br>
2.BERT模型 https://blog.csdn.net/weixin_41519463/article/details/100863313?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-100863313-blog-103734611.235%5Ev38%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-100863313-blog-103734611.235%5Ev38%5Epc_relevant_default_base3&utm_relevant_index=2 <br>
