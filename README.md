# yolov8-face-landmarks-opencv-dnn
使用OpenCV部署yolov8检测人脸和关键点，包含C++和Python两个版本的程序，只依赖opencv库就可以运行。


训练源码是https://github.com/derronqi/yolov8-face
如果想做车牌检测4个角点，那就把检测5个人脸关键点改成4个

此外，添加了人脸质量评估模型fqa，需要结合人脸检测来使用，对应的程序是main_fqa.py和main_fqa.cpp
