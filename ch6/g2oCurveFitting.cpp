#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

// 顶点：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
  // Eigen库为了使用SSE加速，在内存上分配了128位的指针。但是在生成定长的Matrix或Vector对象时，
  // 需要开辟内存，调用默认构造函数，通常x86下的指针是32位，内存位数没对齐就会导致程序运行出错；
  // 对于动态变量(例如Eigen::VectorXd)会动态分配内存，因此会自动地进行内存对齐。
  // 该问题在编译时不会报错，只在运行时报错。
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 重置：顶点初始化
  virtual void setToOriginImpl() override { _estimate << 0, 0, 0; }

  // 更新：参考手写高斯牛顿法的这句话 ae += dx[0] 也就是用 Δx 来更新 abc
  virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update);
  }

  // 存盘和读盘：留空
  // 这里我加了 return 只是为了编译时不报警告，强迫症而已。
  virtual bool read(istream &in) { return 0; }
  virtual bool write(ostream &out) const { return 0; }
};

// 边（误差）：观测值维度，类型，连接顶点类型
class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

  // 计算曲线模型误差 e
  virtual void computeError() override {
    // 本次图优化只有一个节点；
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);
    // 返回当前的预测值
    const Eigen::Vector3d abc = v->estimate();
    // _measurement 是真实值（带噪声）
    _error(0, 0) = _measurement -
                   std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  }

  // 手动指定雅可比矩阵 de/da, de/db, de/dc
  virtual void linearizeOplus() override {
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    // 预测值
    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    _jacobianOplusXi[0] = -_x * _x * y;
    _jacobianOplusXi[1] = -_x * y;
    _jacobianOplusXi[2] = -y;
  }

  // 这里我加了 return 只是为了编译时不报警告，强迫症而已。
  virtual bool read(istream &in) { return 0; }
  virtual bool write(ostream &out) const { return 0; }

 public:
  double _x;  // x 值， y 值为 _measurement
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;   // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;  // 估计参数值
  int N = 100;                           // 数据点
  double w_sigma = 1.0;                  // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;  // OpenCV随机数产生器

  vector<double> x_data, y_data;  // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) +
                     rng.gaussian(w_sigma * w_sigma));
  }

  // 构建图优化，先设定g2o
  // 每个误差项优化变量（abc）维度为3，误差值（e）维度为1
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
  // 线性求解器类型
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;  // 图模型
  optimizer.setAlgorithm(solver);  // 设置求解器
  optimizer.setVerbose(true);      // 打开调试输出

  // 往图中增加顶点
  CurveFittingVertex *v = new CurveFittingVertex();
  // 设定待优化变量
  v->setEstimate(Eigen::Vector3d(ae, be, ce));
  v->setId(0);  // 指定序号
  optimizer.addVertex(v);

  // 往图中增加边。N组数据就有N个误差项，也就有N条边
  for (int i = 0; i < N; i++) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);            // 把第0个节点和节点v连起来
    edge->setMeasurement(y_data[i]);  // 观测数值
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 /
                         (w_sigma * w_sigma));  // 信息矩阵：协方差矩阵之逆
    optimizer.addEdge(edge);
  }

  // 执行优化
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出优化值
  Eigen::Vector3d abc_estimate = v->estimate();
  cout << "estimated model: " << abc_estimate.transpose() << endl;

  return 0;
}