#ifndef GNSS_PSR_DOPP_FACTOR_H_
#define GNSS_PSR_DOPP_FACTOR_H_

#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <gnss_comm/gnss_constant.hpp>
#include <gnss_comm/gnss_utility.hpp>

#define PSR_TO_DOPP_RATIO                   5

using namespace gnss_comm;

/* 
**  parameters[0]: position and orientation at time k
**  parameters[1]: velocity and acc/gyro bias at time k
**  parameters[2]: position and orientation at time k+1
**  parameters[3]: velocity and acc/gyro bias at time k+1
**  parameters[4]: receiver clock bias in light travelling distance (m)
**  parameters[5]: receiver clock bias change rate in clock bias light travelling distance per second (m/s)
**  parameters[6]: yaw difference between ENU and local coordinate (rad)
**  parameters[7]: anchor point's ECEF coordinate
**  
 */
/**
 * 2：残差是2维，包括伪距和多普勒测量残差
 * 7：k时刻位置p和旋转q，一共7维
 * 9：k时刻速度、加速度计和陀螺仪的9维偏置
 * 7：k+1时刻位置p和旋转q，一共7维
 * 9：k+1时刻速度、加速度计和陀螺仪的9维偏置
 * 1：接收机时钟钟差
 * 1：接收机时钟钟差变化率
 * 1：ENU系和local world系之间的偏航角
 * 3：anchor point在ECEF下的3D位置
 * **/
class GnssPsrDoppFactor : public ceres::SizedCostFunction<2, 7, 9, 7, 9, 1, 1, 1, 3>
{
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GnssPsrDoppFactor() = delete;
        GnssPsrDoppFactor(const ObsPtr &_obs, const EphemBasePtr &_ephem, std::vector<double> &_iono_paras, 
            const double _ratio);
        virtual bool Evaluate1(double const *const *parameters, double *residuals, double **jacobians) const;
        // 使用symbolic expression
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
        bool check_gradients(const std::vector<const double*> &parameters) const;
    private:
        const ObsPtr obs;
        const EphemBasePtr ephem;
        const std::vector<double> &iono_paras;
        double ratio;
        int freq_idx;
        double freq;
        Eigen::Vector3d sv_pos;
        Eigen::Vector3d sv_vel;
        double svdt, svddt, tgd;
        double pr_uura, dp_uura;
        double relative_sqrt_info;
};

#endif