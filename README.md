# yolov5_ncnn_ubunt

## Requirements

  ubuntu14.04 16.04  

  ncnn  

  cmake 

  opencv

## steps:

  1: get yolov5.bin yolov5.params   and edit file path in yolov5.cpp line 157-158

  2: edit anchors in yolov5.cpp line 43-44-45

  3: in your ncnn install location, edit ~/ncnn/examples/CMakeLists.txt(add this code-ncnn_add_example(yolov5)) , and put uolov5.cpp in ~/ncnn/examples/ 

  4: make 

:
## how to detect: use this command and put images in ~/ncnn/build/examples/images/  
 
   cd ~/ncnn/build/examples/   
 
   mkdir images  
 
   ./yolov5 ./images/

    then ncnn will detect all images/*.jpg , print results and save results txt.
 
