#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
// 保证和 opencv3 的兼容性，比如 CV_LOAD_IMAGE_COLOR
#include <opencv2/imgcodecs/legacy/constants_c.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  string first_file = "./1.png";
  string second_file = "./2.png";

  // CV_LOAD_IMAGE_COLOR 在 OpenCV4 中已经被删去，修改有两种方法：
  // 1. 引入头文件 opencv2/imgcodecs/legacy/constants_c.h
  // 2. 改成 cv::IMREAD_UNCHANGED
  Mat img_1 = imread(first_file, CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(second_file, CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  // 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  // 假设有N个特征点，则有N行，每行的内容为BRIEF描述子。
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步：检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步：根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1;
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);
  waitKey(0);

  //-- 第三步：对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步：匹配点对筛选。计算最小距离和最大距离。
  // 泛型算法：返回 pair 类型，分别指向 min 和 max 的迭代器
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) {
                                  return m1.distance < m2.distance;
                                });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max distance : %f \n", max_dist);
  printf("-- Min distance : %f \n", min_dist);

  // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。
  // 但有时候最小距离会非常小，那么大部分匹配对距离都比它大，此时使用经验值30。
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步：绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
              img_goodmatch);
  imshow("all matches", img_match);
  waitKey(0);
  imshow("good matches", img_goodmatch);
  imwrite("matches_cv.png", img_goodmatch);
  waitKey(0);

  return 0;
}
