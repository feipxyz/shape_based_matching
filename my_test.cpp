#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "cxxopts/include/cxxopts.hpp"


// #include "cuda_icp/icp.h"

using namespace std;
using namespace cv;


class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

void angle_test(const string& image_path, const string& config_path,  string mode = "test", bool use_rot = true){
    rapidjson::Document config;
    std::ifstream ifs(config_path);
    rapidjson::IStreamWrapper isw(ifs);
    config.ParseStream(isw);
    assert(config.IsObject());

    line2Dup::Detector detector(128, {4, 8});

    if(mode != "test"){
        Mat img = imread(image_path);
        assert(!img.empty() && "check your img path");
        assert(config.HasMember("roi") && config["roi"].IsArray());

        auto array = config["roi"].GetArray();
        Rect roi(array[0].GetInt(), array[1].GetInt(), array[2].GetInt(), array[3].GetInt());
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = config["padding"].GetInt();
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
        shapes.angle_range = {config["angle_start"].GetFloat(), config["angle_end"].GetFloat()};
        shapes.angle_step = config["angle_step"].GetFloat();

        auto scale_array = config["scale"].GetArray();
        if (scale_array.Size() == 1) {
            shapes.scale_range = {scale_array[0].GetFloat()}; // support just one
        } else if (scale_array.Size() == 2) {
            shapes.scale_range = {scale_array[0].GetFloat(), scale_array[1].GetFloat()}; // support just one
        } else {
            std::cout << "scale range error" << std::endl;
            exit(-1);
        }
        // if scale_array.C

        shapes.scale_step = config["scale_step"].GetFloat(); // support just one
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = config["class_id"].GetString();

        bool is_first = true;

        // for other scales you want to re-extract points: 
        // set shapes.scale_range then produce_infos; set is_first = false;

        int first_id = 0;
        float first_angle = 0;
        for(auto& info: shapes.infos){
            Mat to_show = shapes.src_of(info);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id;

            if(is_first){
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
                first_id = templ_id;
                first_angle = info.angle;

                if(use_rot) is_first = false;
            }else{
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle-first_angle,
                                                {shapes.src.cols/2.0f, shapes.src.rows/2.0f});
            }

            auto templ = detector.getTemplates(class_id, templ_id);
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(to_show, {feat.x+templ[0].tl_x, feat.y+templ[0].tl_y}, 3, {0, 0, 255}, -1);
            }
            
            // will be faster if not showing this
            imshow("train", to_show);
            waitKey(1);

            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        std::string class_path = config["class_path"].GetString();
        std::string info_path = config["info_path"].GetString();
        detector.writeClasses(class_path);
        shapes.save_infos(infos_have_templ, info_path);
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        std::string class_path = config["class_path"].GetString();
        detector.readClasses(ids, class_path);

        // angle & scale are saved here, fetched by match id
        std::string template_path = config["info_path"].GetString();
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(template_path);

        Mat test_img = imread(image_path);
        assert(!test_img.empty() && "check your img path");

        int padding = config["padding"].GetInt();
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = config["stride"].GetInt();
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

//        cvtColor(img, img, CV_BGR2GRAY);

        std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;

        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out();

        if(img.channels() == 1) cvtColor(img, img, COLOR_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 1;
        if(top5>matches.size()) top5=matches.size();

        // nms 
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> idxs2;
        for(auto match: matches){
            cv::Rect box;
            box.x = match.x;
            box.y = match.y;
            auto templ = detector.getTemplates("test", match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs2);

                // for(int i=top5-1; i>=0; i--)
        auto array = config["roi"].GetArray();
        for (auto i : idxs2)
        {
            auto match = matches[i];

            // 270 is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training
            float r_scaled = array[2].GetInt() / 2.0f * infos[match.template_id].scale;
            auto templ = detector.getTemplates("test", match.template_id);
            
            int padding = config["padding"].GetInt();
            float train_img_half_width = array[2].GetInt()/2.0f + padding;
            float train_img_half_height = array[3].GetInt()/2.0f + padding;

            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_height;

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
            }

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

            cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos[match.template_id].angle);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++){
                int next = (i+1==4) ? 0 : (i+1);
                cv::line(img, vertices[i], vertices[next], randColor, 2);
            }

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        cv::Mat drawed = padded_img;
 

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

int main(int argc, char** argv){
    cxxopts::Options options("shape based match", "One line description of shape based match");
    options.add_options()
    ("c, config_file", "config file", cxxopts::value<std::string>())
    ("i, image_file", "image file", cxxopts::value<std::string>())
    ("m, mode", "train or test", cxxopts::value<std::string>())
    ("h, help", "Print usage")
    ;

    auto opts = options.parse(argc, argv);
    if (opts.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string config_file = opts["config_file"].as<std::string>();
    std::string image_file = opts["image_file"].as<std::string>();
    std::string mode = opts["mode"].as<std::string>();
    angle_test(image_file, config_file, mode);



    // scale_test("test");
    // angle_test("test", true); // test or train
    // noise_test("test");
    return 0;
}
