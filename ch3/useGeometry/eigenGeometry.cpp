#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// 本程序演示了 Eigen 几何模块的使用方法
// Eigen/Geometry 模块提供了各种旋转和平移的表示

int main(int argc, char **argv)
{
     // Part 1: 旋转矩阵和旋转向量
     // 3D 旋转矩阵直接使用 Matrix3d 或 Matrix3f
     Matrix3d rotation_matrix = Matrix3d::Identity(); // 初始化为单位阵
     // 旋转向量使用 AngleAxis, 它底层不直接是 Matrix，但运算可以当作矩阵（因为重载了运算符）
     AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1)); //沿 Z 轴旋转 45 度

     cout.precision(3);

     // 旋转向量->旋转矩阵。两种方式：matrix(), toRatationMatrix()
     cout << "rotation matrix =\n"
          << rotation_vector.matrix() << endl;
     rotation_matrix = rotation_vector.toRotationMatrix();

     // 旋转向量和旋转矩阵的使用
     Vector3d v(1, 0, 0);
     Vector3d v_rotated = rotation_vector * v; // 旋转向量
     cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
     v_rotated = rotation_matrix * v; // 旋转矩阵
     cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

     /* ------------------------------------------ */

     // Part 2: 欧拉角
     // 旋转矩阵->欧拉角
     Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX 顺序：yaw-pitch-roll（偏航-俯仰-滚转）
     cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

     /* ------------------------------------------ */

     // Part 3: 欧氏变换矩阵
     Isometry3d T = Isometry3d::Identity(); // 虽然称为 3d，实质上是 4＊4 的矩阵
     T.rotate(rotation_vector);             // 按照rotation_vector进行旋转
     T.pretranslate(Vector3d(1, 3, 4));     // 把平移向量设成(1,3,4)
     cout << "Transform matrix = \n"
          << T.matrix() << endl;

     // 变换矩阵的使用
     Vector3d v_transformed = T * v; // 相当于R*v+t
     cout << "v tranformed = " << v_transformed.transpose() << endl;

     // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略

     /* ------------------------------------------ */

     // Part 3: 四元数
     // 可以直接用旋转向量和旋转矩阵初始化四元数，反之亦然
     Quaterniond q = Quaterniond(rotation_vector);
     // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
     cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
     q = Quaterniond(rotation_matrix);
     cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;

     // 四元数的使用：下面的乘法是重载后的运算符
     v_rotated = q * v; // 注意数学上是qvq^{-1}
     cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
     // 用常规向量乘法：qvq^{-1}，则应该如下计算
     cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

     return 0;
}
