#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

/// 本程序演示sophus的基本用法

int main(int argc, char **argv) {
  // 旋转向量（沿Z轴转90度）->旋转矩阵
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  // 旋转矩阵->四元数
  Quaterniond q(R);

  // 1. SO3 可以直接从旋转矩阵和四元数构造
  Sophus::SO3d SO3_R(R);
  Sophus::SO3d SO3_q(q);
  // 二者是等价的
  cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
  cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "they are equal" << endl << endl;

  // 2. 使用对数映射获得它的李代数
  Vector3d so3 = SO3_R.log();
  // 观察一下会发现：so3正是旋转向量
  cout << "so3 = " << so3.transpose() << endl;
  // hat 为向量到反对称矩阵
  cout << "so3 hat（向量->反对称矩阵）=\n" << Sophus::SO3d::hat(so3) << endl;
  // 相对的，vee为反对称到向量
  cout << "so3 hat vee（反对称矩阵->向量）= "
       << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl << endl;

  // 3. 增量扰动模型的更新
  Vector3d update_so3(1e-4, 0, 0);  //假设更新量为这么多；
  // 调用指数映射的方法：SO3d::exp(so3)，和对数映射的调用方式不太相同；
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

  cout << "*******************************" << endl;

  // 4. 构造 SE(3)
  Vector3d t(1, 0, 0);        // 沿X轴平移1
  Sophus::SE3d SE3_Rt(R, t);  // 从R,t构造SE(3)
  Sophus::SE3d SE3_qt(q, t);  // 从q,t构造SE(3)
  cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
  cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;

  // 5. SE(3)->se(3) 
  // se(3) 是一个六维向量，方便起见先 typedef 一下
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  // 观察输出会发现：在Sophus中，se(3)的平移在前，旋转在后；
  // 平移向量是经过线性变换后的结果；旋转向量则是so3
  cout << "se3 = " << se3.transpose() << endl;

  // 6. 同样的，有 hat 和 vee 两个算符
  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = "
       << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl << endl;

  // 7. 左乘扰动
  Vector6d update_se3;  // 扰动
  update_se3.setZero();
  update_se3(0, 0) = 1e-4; // 把第1行第1个元素（平移部分）设为1e-4，
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}
