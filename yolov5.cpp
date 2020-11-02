#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <fstream>
#include <stdlib.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

typedef struct{
        int width;
        int height;
    }Size;

typedef struct {
    std::string name;
    int stride;
    std::vector<cv::Size> anchors;
}YoloLayerData;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

int input_size = 640;
int num_class = 80;
std::vector<YoloLayerData> layers{
        {"547",32,{{116,90},{156,198},{373,326}}},
        {"527",16,{{30,61},{62,45},{59,119}}},
        {"output",8,{{10,13},{16,30},{33,23}}}, 
};

float fast_exp(float x)
{
    union {uint32_t i;float f;} v{};
    v.i=(1<<23)*(1.4426950409*x+126.93490512f);
    return v.f;
}

float sigmoid(float x){
    return 1.0f / (1.0f + fast_exp(-x));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

std::vector<BoxInfo> decode_infer(ncnn::Mat &data, int stride, const cv::Size &frame_size, int net_size_w, int net_size_h, int num_classes,const std::vector<cv::Size>& anchors, float threshold, int x, int y) {
    std::vector<BoxInfo> result;
    int grid_size = int(sqrt(data.h));
    float *mat_data[data.c];
    for(int i=0;i<data.c;i++){
        mat_data[i] = data.channel(i);
    }
    float cx,cy,w,h;
    for(int shift_y=0;shift_y<grid_size;shift_y++){
        for(int shift_x=0;shift_x<grid_size;shift_x++){
            int loc = shift_x+shift_y*grid_size;
            for(int i=0;i<3;i++){
                float *record = mat_data[i];
                //float *record = data.channel(i);
                float *cls_ptr = record + 5;
                for(int cls = 0; cls<num_classes;cls++){
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    if(score>threshold && score <= 0.9999999){
                        cx = (sigmoid(record[0]) * 2.f - 0.5f + (float)shift_x) * (float) stride;
                        cy = (sigmoid(record[1]) * 2.f - 0.5f + (float)shift_y) * (float) stride;
                        w = pow(sigmoid(record[2]) * 2.f,2)*anchors[i].width;
                        h = pow(sigmoid(record[3]) * 2.f,2)*anchors[i].height;
                        BoxInfo box;
                        box.x1 = std::max(0,std::min(frame_size.width,int((cx - w / 2.f - x) * (float)frame_size.width / (float)net_size_w)));
                        box.y1 = std::max(0,std::min(frame_size.height,int((cy - h / 2.f - y) * (float)frame_size.height / (float)net_size_h)));
                        box.x2 = std::max(0,std::min(frame_size.width,int((cx + w / 2.f - x) * (float)frame_size.width / (float)net_size_w)));
                        box.y2 = std::max(0,std::min(frame_size.height,int((cy + h / 2.f - y) * (float)frame_size.height / (float)net_size_h)));
                        box.score = score;
                        box.label = cls;
                        result.push_back(box);
                    }
                }
            }
            for(auto& ptr:mat_data){
                ptr+=(num_classes + 5);
            }
        }
    }
    return result;
}

void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b){return a.score > b.score;});
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


static std::vector<BoxInfo>detect_yolov5(cv::Mat& bgr, float threshold, float nms_threshold, int img_w, int img_h)
{
    ncnn::Net yolov5;
    yolov5.opt.use_vulkan_compute = false;
    yolov5.load_param("/home/pcl/android_work/ncnn/build/examples/yolov5.param");
    yolov5.load_model("/home/pcl/android_work/ncnn/build/examples/yolov5.bin");
    const int target_size = 640;

    float r_w = input_size / (img_w*1.0);
    float r_h = input_size/ (img_h*1.0);
    int true_w, true_h, x, y;
    if (r_h > r_w) {
        true_w = input_size;
        true_h = r_w * img_h;
        x = 0;
        y = (input_size - true_h) / 2;
    } else {
        true_w = r_h* img_w;
        true_h = input_size;
        x = (input_size - true_w) / 2;
        y = 0;
    }
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.set_num_threads(4);
    ex.input(0, in);
    std::vector<BoxInfo> result;
        for(const auto& layer: layers){
            ncnn::Mat blob;
            ex.extract(layer.name.c_str(),blob);
            auto boxes = decode_infer(blob,layer.stride,{img_w, img_h},true_w, true_h,num_class,layer.anchors,threshold, x, y);
            result.insert(result.begin(),boxes.begin(),boxes.end());
        }
    nms(result,nms_threshold);
    printf("finished inference\n");
    return result;
  
}


static void draw_objects_v5(const cv::Mat& bgr, const std::vector<BoxInfo> results, std::string img_name)
{
    static const char* class_names[] = {"person", "bicycle", "car", "motorcycle", "aeroplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign",
                                        "parking meter", "bench", "bird", "cat", "dog", "horse",
                                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase",
                                        "frisbee", "skis", "snowboard", "sports ball", "kite",
                                        "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork",
                                        "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                        "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                        "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                        "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < results.size(); i++)
    {
        const BoxInfo& obj = results[i];

        fprintf(stderr, "%s %.5f %.2f %.2f %.2f %.2f\n", class_names[obj.label], obj.score,
                obj.x1, obj.y1, obj.x2, obj.y2);

        
        cv::rectangle(image, cv::Point(obj.x1, obj.y1), cv::Point(obj.x2, obj.y2), cv::Scalar(255, 0, 0));
        
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.score * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x1 + (obj.x2-obj.x1)/2;
        int y = obj.y1 + (obj.y2-obj.y1)/2 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), 1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("results_ncnn/" + img_name, image);
  	/*cv::namedWindow("img", 0);
  	cv::resizeWindow("img", 1000, 1000);
    cv::imshow("img", image);
    cv::waitKey(0);
    cv::destroyWindow("img");*/
}

static void write_result_txt(const std::vector<BoxInfo> results, std::string img_name)
{
    static const char* class_names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "aeroplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign",
                                        "parking meter", "bench", "bird", "cat", "dog", "horse",
                                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase",
                                        "frisbee", "skis", "snowboard", "sports ball", "kite",
                                        "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork",
                                        "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                        "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                        "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                        "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"
                                       };
    FILE *fp = fopen( ("results/" +  img_name+".txt").c_str(), "w");       
    
    
    for (size_t i = 0; i < results.size(); i++)
    {
        const BoxInfo& obj = results[i];

        fprintf(fp, "%s %.5f %.2f %.2f %.2f %.2f\n", class_names[obj.label], obj.score, obj.x1, obj.y1, obj.x2, obj.y2);
    }

    fclose(fp);
}

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = input_size / (img.cols*1.0);
    float r_h = input_size/ (img.rows*1.0);
    if (r_h > r_w) {
        w = input_size;
        h = r_w * img.rows;
        x = 0;
        y = (input_size - h) / 2;
    } else {
        w = r_h* img.cols;
        h = input_size;
        x = (input_size - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    //printf("orignal w h is %d, %d\n", w, h);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(input_size, input_size, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    const char* imagepath = argv[1];
   /* cv::Mat img = cv::imread(imagepath, 1);
    int img_w = img.cols;
    int img_h = img.rows;
    float conf = 0.3;
    float nms_th = 0.5;
   	//cv::resize(m, m, cv::Size(640, 640));
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    std::vector<BoxInfo> results;
    cv::Mat resize_m = preprocess_img(img);
    results = detect_yolov5(resize_m, conf, nms_th, img_w, img_h);
    draw_objects_v5(img, results);*/
    
    std::vector<std::string> file_names;
    if (read_files_in_dir(imagepath, file_names) < 0) 
    {
      printf("read_files_in_dir failed.");
      return -1;
    }

;

    for (int f = 0; f < (int)file_names.size(); f++) 
    {

        cv::Mat img = cv::imread(std::string(imagepath) + "/" + file_names[f]);
        printf("%s\n", file_names[f].c_str());
        if (img.empty()) continue;
        int img_w = img.cols;
        int img_h = img.rows;
        float conf = 0.3;
        float nms_th = 0.5;
       	//cv::resize(m, m, cv::Size(640, 640));
        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
    
        std::vector<Object> objects;
        std::vector<BoxInfo> results;
        cv::Mat resize_m = preprocess_img(img);
        results = detect_yolov5(resize_m, conf, nms_th, img_w, img_h);
        draw_objects_v5(img, results, file_names[f]);   
        write_result_txt(results, file_names[f]);
        
    }
    return 0;
}
