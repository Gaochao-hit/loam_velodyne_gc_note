// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//这一节点主要功能是：对点云和IMU数据进行预处理，用于特征点的配准。所以这个节点实际上就是一个计算准备的过程，其实就做了一个工作：那就是根据点的曲率c来将点划分为不同的类别

/******************************读前须知*****************************************/
/*imu为x轴向前,y轴向左,z轴向上的右手坐标系，
  velodyne lidar被安装为x轴向前,y轴向左,z轴向上的右手坐标系，
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前,x轴向左,y轴向上的右手坐标系
  ，这是J. Zhang的论文里面使用的坐标系
  交换后：R = Ry(yaw)*Rx(pitch)*Rz(roll)//todo : important
*******************************************************************************/
#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include "utility.h"
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;
using namespace Eigen;

/***********************todo 自己添加与imu 解算相关**************************/
sensor_msgs::Imu  imu_out;

int init_imu_times = 0;

double roll_init = 0.0;
double pitch_init = 0.0;
double yaw_init = 0.0;

double roll_last;double  pitch_last;double yaw_last;
double roll_now; double  pitch_now; double yaw_now;

double angular_x_last = 0.0;
double angular_y_last = 0.0;
double angular_z_last = 0.0;

double angu_x_bias = 0.0;
double angu_y_bias = 0.0;
double angu_z_bias = 0.0;
double angu_x_bias_sum = 0.0;
double angu_y_bias_sum = 0.0;
double angu_z_bias_sum = 0.0;

double linear_acc_sum_x = 0.0;
double linear_acc_sum_y = 0.0;
double linear_acc_sum_z = 0.0;

double linear_acc_x = 0.0;
double linear_acc_y = 0.0;
double linear_acc_z = 0.0;

double imu_time_last;
bool imu_initilized = false;
Eigen::Quaterniond q_start(0,0,0,1);
Eigen::Quaterniond q_end;

/************************todo 结束   ***************************************/

const double scanPeriod = 0.1;//扫描周期，velodyne频率10Hz，周期0.1s  todo:此处可以将该参数放在launch文件，采用参数解析的方式给它
//初始化控制变量
const int systemDelay = 35;//弃用前20帧数据   原值20
int systemInitCount = 0;// 计算弃用的数据的帧数
bool systemInited = false;

const int N_SCANS = 16;//激光雷达的线数   todo： 此处可以修改接口

float cloudCurvature[40000];//点云曲率  40000为一帧点云中点的最大数量
int cloudSortInd[40000];    //曲率点对应的序号  用于曲率排序
int cloudNeighborPicked[40000];//点是否筛选过标志：0-未筛选过，1-筛选过(可以用于确定那些点)
int cloudLabel[40000];//点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)

int imuPointerFront = 0;//imu 时间戳大于当前点云时间戳的位置
int imuPointerLast = -1;//imu最新收到的点在数组中的位置   imuPointerFront与imuPointerLast联合使用可以进行点云的位置插值处理
const int imuQueLength = 200;//imu循环队列长度

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;//点云数据开始的第一个点的欧拉角
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;//点云数据开始的第一个点的速度、位移
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;
//每次点云数据当前点相对于开始第一个点的畸变位移，速度
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;
//IMU 信息
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};//imu在世界系下的平移
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;




/**************************todo 自己****************************************/
Quaterniond euler2quaternion(Vector3d euler)
{
    double cr = std::cos(euler(0)/2);
    double sr = sin(euler(0)/2);
    double cp = cos(euler(1)/2);
    double sp = sin(euler(1)/2);
    double cy = cos(euler(2)/2);
    double sy = sin(euler(2)/2);
    Quaterniond q;
    q.w() = cr*cp*cy + sr*sp*sy;
    q.x() = sr*cp*cy - cr*sp*sy;
    q.y() = cr*sp*cy + sr*cp*sy;
    q.z() = cr*cp*sy - sr*sp*cy;
    return q;
}


int imu_integration()
{
    if(!imu_initilized)//初始化
    {
        if( init_imu_times < 250)//需要静置1秒钟，计算陀螺仪零偏
        {
            init_imu_times++;
            angu_x_bias_sum += imu_out.angular_velocity.x;
            angu_y_bias_sum += imu_out.angular_velocity.y;
            angu_z_bias_sum += imu_out.angular_velocity.z;

            linear_acc_sum_x += imu_out.linear_acceleration.x;
            linear_acc_sum_y += imu_out.linear_acceleration.y;
            linear_acc_sum_z += imu_out.linear_acceleration.z;
            return 0;
        }
        angu_x_bias = angu_x_bias_sum/250.0;
        angu_y_bias = angu_y_bias_sum/250.0;
        angu_z_bias = angu_z_bias_sum/250.0;
        std::cout<< angu_x_bias << std::endl;

        //求解初始roll,pitch
        //1.求得各轴的加速度测量值
        double init_acc_x = linear_acc_sum_x/250.0;
        double init_acc_y = linear_acc_sum_y/250.0;
        double init_acc_z = linear_acc_sum_z/250.0;
        //计算初始yaw,pitch
        roll_init = std::asin(init_acc_x/9.30);
        pitch_init = std::atan2(-init_acc_y,init_acc_z);
        Eigen::Vector3d rpy(roll_init,pitch_init,yaw_init);
        //初始姿态的四元数表示
        q_start = euler2quaternion(rpy);
        ROS_INFO_STREAM("quartenion: 1.1"  << q_start.normalized().x() << q_start.normalized().y() << q_start.normalized().z() << q_start.normalized().w())  ;


        angular_x_last = imu_out.angular_velocity.x;
        angular_y_last = imu_out.angular_velocity.y;
        angular_z_last = imu_out.angular_velocity.z;

        imu_time_last = imu_out.header.stamp.toSec();
        imu_initilized = true;

    }
    else{
        /*********************陀螺仪积分求解姿态****************************/
        // 陀螺仪计算角度变化量
        double angular_x_now = imu_out.angular_velocity.x;
        double angular_y_now = imu_out.angular_velocity.y;
        double angular_z_now = imu_out.angular_velocity.z;

        double imu_time_now = imu_out.header.stamp.toSec();
        double d_t = imu_time_now - imu_time_last;
        //std::cout << d_t << std::endl;

        double d_angule_x = d_t * ( angular_x_last + angular_x_now - 2 * angu_x_bias)/2.0;//后面此处添加bias
        double d_angule_y = d_t * ( angular_y_last + angular_y_now - 2 * angu_y_bias)/2.0;
        double d_angule_z = d_t * ( angular_z_last + angular_z_now - 2 * angu_z_bias)/2.0;
        //此处角度叠加
        q_end = q_start * Eigen::Quaterniond(1,d_angule_x,d_angule_y,d_angule_z);//陀螺仪在上一次结果上叠加积分结果
        q_end.normalize();
        double angu_roll, angu_pitch, angu_yaw;//角速度积分得到的角度值
        tf::Matrix3x3(tf::Quaternion(q_end.x(),q_end.y(),q_end.z(),q_end.w())).getRPY(angu_roll, angu_pitch, angu_yaw);//todo 陀螺仪计算得到的角度
        //std::cout<< q_end.x() << q_end.y() << q_end.z() << q_end.w() << std::endl;
        /**********************陀螺仪积分结束***************************************/

        imu_out.orientation.x = q_end.normalized().x();
        imu_out.orientation.y = q_end.normalized().y();
        imu_out.orientation.z = q_end.normalized().z();
        imu_out.orientation.w = q_end.normalized().w();
        //imu_out.header.frame_id = "base_link";

        /***************加速度求解imu姿态**********************/
        linear_acc_x = imu_out.linear_acceleration.x;
        linear_acc_y = imu_out.linear_acceleration.y;
        linear_acc_z = imu_out.linear_acceleration.z;
        std::cout << "acc_x: " << linear_acc_x << std::endl;

        double k = 0.95;//融合系数
        double roll_acc = 0.0 ;
        double pitch_acc = 0.0;
        if( abs(linear_acc_x) < 3  &&    abs(linear_acc_y) < 3 )
        {
            roll_acc = std::asin(linear_acc_x/9.80);//todo 加速度计算得到的角度    
            pitch_acc = std::atan2(-linear_acc_y,linear_acc_z);

        }
        else
        {
            roll_acc = angu_roll;
            pitch_acc = angu_pitch;
        }

        //double pitch_acc = std::atan2(-linear_acc_y,linear_acc_z);
        /*************************加速度计算角度结束****************************/

        /*******************两种方法计算出来的角度融合  todo:互补滤波 ********************/


        double fused_roll = k * angu_roll  +  (1 - k) * roll_acc;//todo 此处的roll_acc不稳定
        std::cout << "fused_roll" << angu_roll << "  " << roll_acc <<std::endl;

        double fused_pitch = k *angu_pitch +  (1 - k) * pitch_acc;
        std::cout << "fused_pitch" << angu_pitch << " " << pitch_acc <<std::endl;

        double fused_yaw = angu_yaw;//yaw角没有融合

        std::cout << "fused_yaw" << angu_yaw;

        Eigen::Vector3d rpy(fused_roll,fused_pitch,fused_yaw);
        std::cout << "rpy:  " << fused_roll << fused_pitch << fused_yaw <<std::endl;
        Quaterniond q_fused = euler2quaternion(rpy);
        std::cout << "q_fused:  " << q_fused.x() << q_fused.y() << q_fused.z() << q_fused.w() << std::endl;

        q_fused.normalize();

        std::cout<<"quart3：  " << q_fused.x() << q_fused.y() << q_fused.z() << q_fused.w() << std::endl;

        imu_out.orientation.x = q_fused.x();
        imu_out.orientation.y = q_fused.y();
        imu_out.orientation.z = q_fused.z();
        imu_out.orientation.w = q_fused.w();

        std::cout<<"quart3：  " << imu_out.orientation.x << imu_out.orientation.y << imu_out.orientation.z << imu_out.orientation.w << std::endl;



        //Imu_pub.publish(imu_out);

        //数据转换存储
        imu_time_last = imu_time_now;
        q_start = q_fused;
        angular_x_last = angular_x_now;
        angular_y_last = angular_y_now;
        angular_z_last = angular_z_now;
    }


}
/*************************todo 结束****************************************/


//todo 计算局部坐标系下(imu start系下 )  点云中的点相对第一个开始点的由于加减速运动产生的位移畸变
void ShiftToStartIMU(float pointTime)
{//计算相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;
//绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;
//绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}
//计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）
void VeloToStartIMU()
{
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
//绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;
//绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}
//去除点云加减速产生的位移畸变
void TransformToStartIMU(PointType *p)
{
    //Ry*Rx*Rz*Pl, transform point to the global frame
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;//绕z轴旋转(imuRollCur)
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  float x2 = x1;//绕x轴旋转(imuPitchCur)
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;
//绕y轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;
// Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    // transfrom global points to the local frame
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;//绕y轴旋转(-imuYawStart)
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;//绕x轴旋转(-imuPitchStart)
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;//绕z轴旋转(-imuRollStart)，然后叠加平移量
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}
//积分位移与速度
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];//todo:取出的roll,pitch,yaw是相对于世界系的
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];
//将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，TODO 转换得到世界坐标系下的加速度值(right hand rule)
  //绕z周旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
//绕x轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
//绕y轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;//在世界系下的加速度
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;
//上一个imu点
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];// 相邻两个imu时间点的差值，也就是imu测量周期
  if (timeDiff < scanPeriod) {
//匀加速度模型
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff //求得imu当前在世界坐标系下的位移(即世家坐标系下的坐标）
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;//求得imu在世界坐标系下的速度
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}
//接收点云数据，velodyne雷达坐标系安装为x轴向前，y轴向左，z轴向上的右手坐标系
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)//主要功能是对接收到的点云进行预处理，完成分类。具体分类内容为：一是将点云划入不同线中存储；二是对其进行特征分类。
{
  if (!systemInited) {
    systemInitCount++;
    if (systemInitCount >= systemDelay) {//延迟20帧启动,保证有imu数据后在调用laserCloudHandler
      systemInited = true;
    }
    return;
  }
//记录每个scan(即每条线)有曲率点的开始和结束索引
  std::vector<int> scanStartInd(N_SCANS, 0);//N_SCANS表示激光雷达线数
  std::vector<int> scanEndInd(N_SCANS, 0);
  //当前点云时间
  double timeScanCur = laserCloudMsg->header.stamp.toSec();//雷达时间戳
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);//转为为pcl数据存储
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);////去除无效值(Removes points with x, y, or z equal to NaN)  cloud_out.points[i] = cloud_in.points[index[i]]
  int cloudSize = laserCloudIn.points.size();//点云点的数量
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);//lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,//todo 个人理解是相当于求解与-X轴的夹角
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;
    //结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
//正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  bool halfPassed = false;//lidar扫描线是否旋转过半
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);//遍历所有点，根据角度计算结果将其划分为不同的线：计算角度-->计算起始和终止位置-->插入IMU数据-->将点插入容器中
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn.points[i].y;//此处将雷达系的点向(x轴向前，y轴向左，z轴向上的右手坐标系)转换
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;//计算每个点的仰角，根据仰角排列激光线号   atan函数返回的是弧度
    int scanID;
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); //仰角四舍五入(加减0.5截断效果等于四舍五入)
    if (roundedAngle > 0){//若大于0，则输出角度分别为1,3,5,7,9,11...
      scanID = roundedAngle;//对应的线的标号分别为1,3,5,7,9,11...
        //std::cout<< scanID <<std::endl; //此处自己打印的结果表明自己是对的
    }
    else {//若小于0，则roundedAngle分别是-1,-3,-5...
      scanID = roundedAngle + (N_SCANS - 1);//角度大于零，由小到大划入偶数线（0->16）；角度小于零，由大到小划入奇数线(15->1)  todo : 此处的别人的解释与自己猜想相反，个人感觉角度大于0，划入技奇数线，角度小于0，划入偶数线
    }
    if (scanID > (N_SCANS - 1) || scanID < 0 ){//过滤点，只挑选[-15度，+15度]范围内的点,scanID属于[0,15]
      count--;//将16线以外的杂点剔除,count是点云数量计数，若是杂点，则数量减一
      continue;
    }

    float ori = -atan2(point.x, point.z);//计算每个点的相对方位角(即水平面内的旋转角度)
    if (!halfPassed) {//todo 根据扫描线是否旋转过半选择与起始位置还是终止位置进行插值计算，从而进行补偿
      if (ori < startOri - M_PI / 2) {//确保-pi/2 < ori - startOri < 3*pi/2
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;//过半
      }
    } else {
      ori += 2 * M_PI;

      if (ori < endOri - M_PI * 3 / 2) {//确保-3*pi/2 < ori - endOri < pi/2
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }
//点旋转的角度与整个周期旋转的角度的比例，即点云中点的相对时间
    float relTime = (ori - startOri) / (endOri - startOri);
    point.intensity = scanID + scanPeriod * relTime;//todo 整数部分：scan ID(线号)，小数部分：每个点扫描的时间(相对时间)    匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
    // imuPointerLast 是当前点，变量只在imu中改变，设为t时刻
    if (imuPointerLast >= 0) {
      float pointTime = relTime * scanPeriod;//scanPeriod = 0.1s
      while (imuPointerFront != imuPointerLast) {
          //寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront
          //找到ti后的最近一个imu时刻
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {//timeScanCur + pointTime是ti时刻(第i个点扫描的时间;imuPointerFront是ti后一时刻的imu时间,imuPointerBack是ti前一时刻的imu时间
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }
//todo 下面的if语句我感到很奇怪，个人感觉不会发生，因为如果满足上下述条件，则上面的while循环无法跳转出来
        //todo 此处留待验证
      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {//没有找到ti后的最近一个imu时刻的话，此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];
        //std::cout << "occored" << std::endl;
        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {//找到了点云时间戳小于IMU时间戳的IMU位置,则该点必处于imuPointerBack和imuPointerFront之间，据此线性插值，计算点云点的速度，位移和欧拉角
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) //按时间距离计算权重分配比率,也即线性插值
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;//用ti时间前后的imu数据进行插值
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {//TODO 相比于pitch和roll   yaw角变化可能会大很多  此处尚没有完全理解
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }
      if (i == 0) {//如果是第一个点，记住点云起始位置的速度，位移，欧拉角
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {//计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移速度畸变，并对点云中的每个点位置信息重新补偿矫正
        ShiftToStartIMU(pointTime);// 将Lidar位移转到IMU起始坐标系下
        VeloToStartIMU();//// 将Lidar运动速度转到IMU起始坐标系下
        TransformToStartIMU(&point);// 将点坐标转到起始IMU坐标系下
      }
    }
    laserCloudScans[scanID].push_back(point);// 将点按照每一层线，分类压入16个数组中,将每个补偿矫正的点放入对应线号的容器
  }
  cloudSize = count;//获得有效范围内点的数量

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {
    *laserCloud += laserCloudScans[i];//将所有点存入laserCloud中，点按(线序)进行排列
  }
  int scanCount = -1;
  for (int i = 5; i < cloudSize - 5; i++) {// 对所有的激光点一个一个求出在该点前后5个点(10点)的偏差，作为cloudCurvature点云数据的曲率;前后五个点跳过
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;//曲率的计算公式，对应论文公示(1)  todo :此处没有做normalize处理
    cloudSortInd[i] = i;//记录曲率点的索引
    cloudNeighborPicked[i] = 0;//初始时，点全未被筛选过
    cloudLabel[i] = 0;//初始化为less flat点

    if (int(laserCloud->points[i].intensity) != scanCount) {//每一层只有第一个点符合该条件，进入if语句内部
      scanCount = int(laserCloud->points[i].intensity);
      //因为是按照线的序列存储，因此接下来能够得到起始和终止的index；在这里滤除前五个和后五个。
      if (scanCount > 0 && scanCount < N_SCANS) {//注意此处的逻辑
        scanStartInd[scanCount] = i + 5;//记录每一层start index
        scanEndInd[scanCount - 1] = i - 5;//记录每一层 end index
      }
    }
  }
  scanStartInd[0] = 5;//第一个scan曲率点有效点序从第5个开始，最后一个激光线结束点序size-5
  scanEndInd.back() = cloudSize - 5;

  for (int i = 5; i < cloudSize - 6; i++) {
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;//相邻两个点的距离的平方

    if (diff > 0.1) {//前提:两个点之间距离要大于0.1
///*— 针对论文的(b)情况，两向量夹角小于某阈值b时（夹角小就可能存在遮挡），将其一侧的临近6个点设为不可标记为特征点的点 —*/
      /*— 构建了一个等腰三角形的底向量，根据等腰三角形性质，判断X[i]向量与X[i+1]的夹角小于5.732度(threshold=0.1) —*/
      /*— depth1>depth2 X[i+1]距离更近，远侧点标记不特征；depth1<depth2 X[i]距离更近，远侧点标记不特征 —*/
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + //点i深度
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + //点i + 1的深度
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;//将深度较大的点拉回后计算
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;
//边长比也即是弧度值，若小于0.1，说明夹角比较小
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//排除容易被斜面挡住的点
          cloudNeighborPicked[i - 5] = 1;//当某点及其后点间的距离平方大于某阈值a（说明这两点有一定距离），且两向量夹角小于某阈值b时（夹角小就可能存在遮挡）   将其一侧的临近6个点设为不可标记为特征点的点
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }
//    /*— 针对论文的(a)情况，当某点及其后点间的距离平方大于某阈值a（说明这两点有一定距离） ———*/
    /*— 若某点到其前后两点的距离均大于c倍的该点深度，则该点判定为不可标记特征点的点 ———————*/
    /*—（入射角越小，点间距越大，即激光发射方向与投射到的平面越近似水平） ———————————————*/
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {//与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上的点，强烈凸凹点和空旷区域中的某些点，置为筛选过，弃用
      cloudNeighborPicked[i] = 1;
    }
  }


  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    for (int j = 0; j < 6; j++) {// 将每个线等分为六段，分别进行处理（sp、ep分别为各段的起始和终止位置）
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;//sp = (scanEndInd - scanStartInd)/6 + scanStartInd
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;
//曲率大小从小到大排序(冒泡法)
      for (int k = sp + 1; k <= ep; k++) {// 在每一段，都将曲率按照升序排列
        for (int l = k; l >= sp + 1; l--) {
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }
//挑选每个分段的曲率很大和比较大的点
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];//曲率最大的点序
        if (cloudNeighborPicked[ind] == 0 &&//如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {//挑选曲率最大的前2个点放入sharp点集合    ///*—— 筛选特征角点 Corner: label=2; LessCorner: label=1 ————*/
            cloudLabel[ind] = 2;//2表示曲率很大
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);//cornerPointsLessSharp包含了ind为2和ind为1的点
          } else if (largestPickedNum <= 20) {
            cloudLabel[ind] = 1;//1表示曲率较大
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;//设置筛选标志位
          for (int l = 1; l <= 5; l++) {//将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }
//挑选每个分段的曲率很小比较小的点
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }
//将剩余的点（包括之前被排除的点）全部归入平面点中less flat类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }
//由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;//创建点云彩滤波器
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);//设置体素大小
    downSizeFilter.filter(surfPointsLessFlatScanDS);
//less flat点汇总
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

//publish消除非匀速运动畸变后的所有的点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);
//publich消除非匀速运动畸变后的平面点和边沿点
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);
//publich IMU消息,由于循环到了最后，因此是Cur都是代表最后一个点，即最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
    //起始点欧拉角
  imuTrans.points[0].x = imuPitchStart;//需要发布一个topic /imu_trans类型是sensor_msgs::PointCloud2而不是sensor_msgs::Imu
  imuTrans.points[0].y = imuYawStart;//作者用4个pcl::Point XYZ类型的数组来存储IMU的信息，作者pcl用的6，坑神走起
  imuTrans.points[0].z = imuRollStart;
//最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;
//最后一个点相对于第一个点的几遍位移与速度
  imuTrans.points[2].x = imuShiftFromStartXCur;//imuShiftFromStartXCur是局部坐标系下(imu start系下 )的表示
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
    /***********************todo 此处自己添加*********************/
    imu_out.angular_velocity.x = imuIn->angular_velocity.x;
    imu_out.angular_velocity.y = imuIn->angular_velocity.y;
    imu_out.angular_velocity.z = imuIn->angular_velocity.z;

    imu_out.linear_acceleration.x = imuIn->linear_acceleration.x;
    imu_out.linear_acceleration.y = imuIn->linear_acceleration.y;
    imu_out.linear_acceleration.z = imuIn->linear_acceleration.z;
    imu_out.header.stamp = imuIn->header.stamp;

    imu_integration();
    if(!imu_initilized)
        return;

//    imuIn->orientation.x = imu_out.orientation.x;
//    imuIn->orientation.y = imu_out.orientation.y;
//    imuIn->orientation.z = imu_out.orientation.z;
//    imuIn->orientation.w = imu_out.orientation.w;


    /***************************todo 结束*******************************/


  std::cout<< "normal: 1" << std::endl;
    ROS_INFO_STREAM("normal:1 ");

  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imu_out.orientation, orientation);//此imu可以直接读取旋转信息,而不需要积分  todo 此处原本是imuIn->orientation    ----->    imu_out.orientation
    std::cout<<"quaternion_2:  " << imu_out.orientation.x << imu_out.orientation.y << imu_out.orientation.z << imu_out.orientation.w  << std::endl;
    std::cout<<"normal : 1.1" << std::endl;
    ROS_INFO_STREAM("normal:1.1 ");

  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);//imu的orientation是全局的orientation   now : R = Ry(yaw)*Rx(pitch)*Rz(roll).
  //imu为x轴向前,y轴向左,z轴向上的右手坐标系
  //TODO 此处的计算方法是将求得的旋转矩阵对重力(0,9.8,0)做一个旋转。求得重力在各个方向的影响分量
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;//减去重力对imu的影响
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;//TODO accX,accY,accZ是(z轴向前,x轴向左,y轴向上的右手坐标系)坐标系内的表示(lidar与imu系相差固定的坐标变换)
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;//todo 修改imuPointerLast的值，使得297行处判断出有imu数据可以使用

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  //ros::init(argc, argv, "scanRegistration");
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/rslidar_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/zr300_node/imu", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);// todo:/imu_trans 发布的话题的用途？？

  ros::spin();

  return 0;
}

