# yolov5_ncnn_ubunt

first stage: get yolov5.bin yolov5.params

second stage : edit anchors in your datasets

third stage: edit ncnn/examples/CMakeLists.txt  : ncnn_add_example(yolov5)

fourth stage: make 

fiveth stages: cd build/examples/   mkdir images,  ./yolov5 ./images/

		then ncnn will detect images/*.jpg and print results
 
