#include "gnss_psr_dopp_factor.hpp"

// GNSS伪距和多普勒测量因子
GnssPsrDoppFactor::GnssPsrDoppFactor(const ObsPtr &_obs, const EphemBasePtr &_ephem, 
    std::vector<double> &_iono_paras, const double _ratio) 
        : obs(_obs), ephem(_ephem), iono_paras(_iono_paras), ratio(_ratio)
{
    freq = L1_freq(obs, &freq_idx);
    LOG_IF(FATAL, freq < 0) << "No L1 observation found.";

    uint32_t sys = satsys(obs->sat, NULL);
    double tof = obs->psr[freq_idx] / LIGHT_SPEED;
    gtime_t sv_tx = time_add(obs->time, -tof);

    if (sys == SYS_GLO)
    {
        GloEphemPtr glo_ephem = std::dynamic_pointer_cast<GloEphem>(ephem);
        svdt = geph2svdt(sv_tx, glo_ephem);
        sv_tx = time_add(sv_tx, -svdt);
        sv_pos = geph2pos(sv_tx, glo_ephem, &svdt);
        sv_vel = geph2vel(sv_tx, glo_ephem, &svddt);
        tgd = 0.0;
        pr_uura = 2.0 * (obs->psr_std[freq_idx]/0.16);
        dp_uura = 2.0 * (obs->dopp_std[freq_idx]/0.256);
    }
    else
    {
        EphemPtr eph = std::dynamic_pointer_cast<Ephem>(ephem);
        svdt = eph2svdt(sv_tx, eph);
        sv_tx = time_add(sv_tx, -svdt);
        sv_pos = eph2pos(sv_tx, eph, &svdt);
        sv_vel = eph2vel(sv_tx, eph, &svddt);
        tgd = eph->tgd[0];
        if (sys == SYS_GAL)
        {
            pr_uura = (eph->ura - 2.0) * (obs->psr_std[freq_idx]/0.16);
            dp_uura = (eph->ura - 2.0) * (obs->dopp_std[freq_idx]/0.256);
        }
        else
        {
            pr_uura = (eph->ura - 1.0) * (obs->psr_std[freq_idx]/0.16);
            dp_uura = (eph->ura - 1.0) * (obs->dopp_std[freq_idx]/0.256);
        }
    }

    LOG_IF(FATAL, pr_uura <= 0) << "pr_uura is " << pr_uura;
    LOG_IF(FATAL, dp_uura <= 0) << "dp_uura is " << dp_uura;
    relative_sqrt_info = 10.0;
}

bool GnssPsrDoppFactor::Evaluate1(double const *const *parameters, double *residuals, double **jacobians) const
{
    // auto start = std::chrono::steady_clock::now(); // 2024-7-2
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    double rcv_dt = parameters[4][0];
    double rcv_ddt = parameters[5][0];
    double yaw_diff = parameters[6][0];
    Eigen::Vector3d ref_ecef(parameters[7][0], parameters[7][1], parameters[7][2]);

    const Eigen::Vector3d local_pos = ratio*Pi + (1.0-ratio)*Pj;// 因为GNSS数据的时间戳和Image时间戳并不是完全对上的，所以这里应该是使用一个比列插值，得到用于GNSS优化的local_pos
    const Eigen::Vector3d local_vel = ratio*Vi + (1.0-ratio)*Vj;

    double sin_yaw_diff = std::sin(yaw_diff);
    double cos_yaw_diff = std::cos(yaw_diff);
    Eigen::Matrix3d R_enu_local;
    R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                   sin_yaw_diff,  cos_yaw_diff, 0,
                   0           ,  0           , 1;
    Eigen::Matrix3d R_ecef_enu = ecef2rotation(ref_ecef);
    Eigen::Matrix3d R_ecef_local = R_ecef_enu * R_enu_local;

    Eigen::Vector3d P_ecef = R_ecef_local * local_pos + ref_ecef;
    Eigen::Vector3d V_ecef = R_ecef_local * local_vel;

    double ion_delay = 0, tro_delay = 0;
    double azel[2] = {0, M_PI/2.0};
    if (P_ecef.norm() > 0)
    {
        // 计算卫星的方位角/仰角
        sat_azel(P_ecef, sv_pos, azel);
        Eigen::Vector3d rcv_lla = ecef2geo(P_ecef);
        tro_delay = calculate_trop_delay(obs->time, rcv_lla, azel);
        ion_delay = calculate_ion_delay(obs->time, iono_paras, rcv_lla, azel);
    }
    double sin_el = sin(azel[1]);
    double sin_el_2 = sin_el*sin_el;
    double pr_weight = sin_el_2 / pr_uura * relative_sqrt_info;
    double dp_weight = sin_el_2 / dp_uura * relative_sqrt_info * PSR_TO_DOPP_RATIO;

    Eigen::Vector3d rcv2sat_ecef = sv_pos - P_ecef;
    Eigen::Vector3d rcv2sat_unit = rcv2sat_ecef.normalized();

    // 关于前缀SV: SV refers to space vehicle. 即航天器;空间飞行器;宇宙飞船;航天飞行器;太空交通工具
    // SV就是space vehicle，翻译为“空间飞行器”，在GNSS领域，SV指空间中运行的卫星
    // 作者考虑了萨格纳克效应Sagnac Effect, 受地球自转的影响, 但是新版论文却没有提及。沿用自RTKLIB
    const double psr_sagnac = EARTH_OMG_GPS*(sv_pos(0)*P_ecef(1)-sv_pos(1)*P_ecef(0))/LIGHT_SPEED;
    // ion_delay和tro_delay用长度单位表示，即已经乘以了光速c
    // svdt卫星的时钟钟差, rcv_dt接收机时钟钟差，居然也是长度单位，但作者在论文中也没有说明
    double psr_estimated = rcv2sat_ecef.norm() + psr_sagnac + rcv_dt - svdt*LIGHT_SPEED + 
                                ion_delay + tro_delay + tgd*LIGHT_SPEED;
    
    // 伪距测量误差：估计值（预测值）减去测量值
    residuals[0] = (psr_estimated - obs->psr[freq_idx]) * pr_weight;

    const double dopp_sagnac = EARTH_OMG_GPS/LIGHT_SPEED*(sv_vel(0)*P_ecef(1)+
            sv_pos(0)*V_ecef(1) - sv_vel(1)*P_ecef(0) - sv_pos(1)*V_ecef(0));
    // 计算预估的多普勒频移，这里其实换成了速度, 并取了负号，所以后面求残差时变成相加。        
    double dopp_estimated = (sv_vel - V_ecef).dot(rcv2sat_unit) + dopp_sagnac + rcv_ddt - svddt*LIGHT_SPEED;
    const double wavelength = LIGHT_SPEED / freq;
    // 多普勒测量误差
    // obs->dopp[freq_idx]应该表示的是测量到的多普勒频移
    // 这里实际上把多普勒频移乘以了波长，换算成速度来做误差项
    residuals[1] = (dopp_estimated + obs->dopp[freq_idx]*wavelength) * dp_weight;

    if (jacobians)
    {
        // J_Pi
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_Pi(jacobians[0]);
            J_Pi.setZero();
            J_Pi.topLeftCorner<1, 3>() = -rcv2sat_unit.transpose() * R_ecef_local * pr_weight * ratio; // 伪距误差部分对Pi的导数

            const double norm3 = pow(rcv2sat_ecef.norm(), 3);
            const double norm2 = rcv2sat_ecef.squaredNorm();
            Eigen::Matrix3d unit2rcv_pos;
            for (size_t i = 0; i < 3; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    if (i == j)
                        unit2rcv_pos(i, j) = (norm2-rcv2sat_ecef(i)*rcv2sat_ecef(i))/norm3;
                    else
                        unit2rcv_pos(i, j) = (-rcv2sat_ecef(i)*rcv2sat_ecef(j))/norm3;
                }
            }
            unit2rcv_pos *= -1;
            J_Pi.bottomLeftCorner<1, 3>() = (sv_vel-V_ecef).transpose() * unit2rcv_pos * // 多普勒测量部分对Pi的导数
                R_ecef_local * dp_weight * ratio;
        }

        // J_Vi
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_Vi(jacobians[1]);
            J_Vi.setZero();
            J_Vi.bottomLeftCorner<1, 3>() = rcv2sat_unit.transpose() * (-1.0) * 
                R_ecef_local * dp_weight * ratio;
        }

        // J_Pj
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_Pj(jacobians[2]);
            J_Pj.setZero();
            J_Pj.topLeftCorner<1, 3>() = -rcv2sat_unit.transpose() * R_ecef_local * pr_weight * (1.0-ratio);

            const double norm3 = pow(rcv2sat_ecef.norm(), 3);
            const double norm2 = rcv2sat_ecef.squaredNorm();
            Eigen::Matrix3d unit2rcv_pos;
            for (size_t i = 0; i < 3; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    if (i == j)
                        unit2rcv_pos(i, j) = (norm2-rcv2sat_ecef(i)*rcv2sat_ecef(i))/norm3;
                    else
                        unit2rcv_pos(i, j) = (-rcv2sat_ecef(i)*rcv2sat_ecef(j))/norm3;
                }
            }
            unit2rcv_pos *= -1;
            J_Pj.bottomLeftCorner<1, 3>() = (sv_vel-V_ecef).transpose() * unit2rcv_pos * 
                R_ecef_local * dp_weight * (1.0-ratio);
        }

        // J_Vj
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_Vj(jacobians[3]);
            J_Vj.setZero();
            J_Vj.bottomLeftCorner<1, 3>() = rcv2sat_unit.transpose() * (-1.0) * 
                R_ecef_local * dp_weight * (1.0-ratio);
        }

        // J_rcv_dt
        if (jacobians[4])
        {
            jacobians[4][0] = 1.0 * pr_weight;
            jacobians[4][1] = 0;
        }

        // J_rcv_ddt
        if (jacobians[5])
        {
            jacobians[5][0] = 0;
            jacobians[5][1] = 1.0 * dp_weight;
        }

        // J_yaw_diff
        if (jacobians[6])
        {
            Eigen::Matrix3d d_yaw;
            d_yaw << -sin_yaw_diff, -cos_yaw_diff, 0, 
                      cos_yaw_diff, -sin_yaw_diff, 0, 
                      0           ,  0           , 0;
            jacobians[6][0] = -rcv2sat_unit.dot(R_ecef_enu * d_yaw * local_pos) * pr_weight;
            jacobians[6][1] = -rcv2sat_unit.dot(R_ecef_enu * d_yaw * local_vel) * dp_weight;
        }

        // J_ref_ecef, approximation for simplicity
        if (jacobians[7])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_ref_ecef(jacobians[7]);
            J_ref_ecef.setZero();
            J_ref_ecef.row(0) = -rcv2sat_unit.transpose() * pr_weight;
        }
    }

    // auto end = std::chrono::steady_clock::now(); // 2024-7-2
    // auto diff = end - start;
    // std::cout << "gnss_psr_dopp_factor: " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    return true;
}

// 实现了用符号表达式来计算residual和jacobian
using Scalar = double;
bool GnssPsrDoppFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // auto start = std::chrono::steady_clock::now(); // 2024-7-2
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    double rcv_dt = parameters[4][0];
    double rcv_ddt = parameters[5][0];
    double yaw_diff = parameters[6][0];
    Eigen::Vector3d ref_ecef(parameters[7][0], parameters[7][1], parameters[7][2]);

    double ion_delay = 0, tro_delay = 0;
    double azel[2] = {0, M_PI/2.0};
    

    ///////////////////////////////////////////////////////////////

    const double psr_measured = obs->psr[freq_idx];
    const double dopp_measured = obs->dopp[freq_idx];
    const double epsilon = 2.22044604925031e-15;

    // Intermediate terms (322)
    const Scalar _tmp0 = Scalar(1.0) - ratio;
    const Scalar _tmp1 = Pi(2, 0) * ratio + Pj(2, 0) * _tmp0;
    const Scalar _tmp2 =
        epsilon * ((((ref_ecef(0, 0)) > 0) - ((ref_ecef(0, 0)) < 0)) + Scalar(0.5)) + ref_ecef(0, 0);
    const Scalar _tmp3 = Scalar(1.0) * std::atan2(ref_ecef(1, 0), _tmp2);
    const Scalar _tmp4 = std::cos(_tmp3);
    const Scalar _tmp5 = [&]() {
        const Scalar base = ref_ecef(2, 0);
        return base * base * base;
    }();
    const Scalar _tmp6 = std::pow(ref_ecef(2, 0), Scalar(2));
    const Scalar _tmp7 = std::pow(ref_ecef(1, 0), Scalar(2));
    const Scalar _tmp8 = _tmp7 + epsilon + std::pow(ref_ecef(0, 0), Scalar(2));
    const Scalar _tmp9 =
        40680631590769 * _tmp6 + Scalar(40408299984661.5) * _tmp8 + std::pow(epsilon, Scalar(2));
    const Scalar _tmp10 = std::pow(_tmp9, Scalar(Scalar(-3) / Scalar(2)));
    const Scalar _tmp11 = Scalar(1.0) / (epsilon + Scalar(40408299984661.5));
    const Scalar _tmp12 = _tmp10 * _tmp11;
    const Scalar _tmp13 = Scalar(4.4917426690159599e+38) * _tmp12 * _tmp5 + ref_ecef(2, 0);
    const Scalar _tmp14 = (_tmp8 * std::sqrt(_tmp8));
    const Scalar _tmp15 = std::sqrt(_tmp8);
    const Scalar _tmp16 = -Scalar(1.09675613733197e+25) * _tmp10 * _tmp14 + _tmp15 + epsilon;
    const Scalar _tmp17 = Scalar(1.0) / (_tmp16);
    const Scalar _tmp18 = Scalar(1.0) * std::atan(_tmp13 * _tmp17);
    const Scalar _tmp19 = _tmp18;
    const Scalar _tmp20 = std::cos(_tmp19);
    const Scalar _tmp21 = _tmp20 * _tmp4;
    const Scalar _tmp22 = Pi(0, 0) * ratio + Pj(0, 0) * _tmp0;
    const Scalar _tmp23 = std::sin(_tmp19);
    const Scalar _tmp24 = std::sin(yaw_diff);
    const Scalar _tmp25 = _tmp24 * _tmp4;
    const Scalar _tmp26 = std::sin(_tmp3);
    const Scalar _tmp27 = std::cos(yaw_diff);
    const Scalar _tmp28 = _tmp26 * _tmp27;
    const Scalar _tmp29 = -_tmp23 * _tmp25 - _tmp28;
    const Scalar _tmp30 = Pi(1, 0) * ratio + Pj(1, 0) * _tmp0;
    const Scalar _tmp31 = _tmp27 * _tmp4;
    const Scalar _tmp32 = _tmp24 * _tmp26;
    const Scalar _tmp33 = -_tmp23 * _tmp31 + _tmp32;
    const Scalar _tmp34 = _tmp1 * _tmp21 + _tmp22 * _tmp29 + _tmp30 * _tmp33 + ref_ecef(0, 0);
    const Scalar _tmp35 = _tmp20 * _tmp26;
    const Scalar _tmp36 = -_tmp23 * _tmp32 + _tmp31;
    const Scalar _tmp37 = -_tmp23 * _tmp28 - _tmp25;
    const Scalar _tmp38 = _tmp1 * _tmp35 + _tmp22 * _tmp36 + _tmp30 * _tmp37 + ref_ecef(1, 0);
    const Scalar _tmp39 = -_tmp34 + sv_pos(0, 0);
    const Scalar _tmp40 = -_tmp38 + sv_pos(1, 0);
    const Scalar _tmp41 = _tmp20 * _tmp27;
    const Scalar _tmp42 = _tmp20 * _tmp24;

    
    const Scalar _tmp4333 = _tmp1 * _tmp23 + _tmp22 * _tmp42 + _tmp30 * _tmp41 + ref_ecef(2, 0);
    Eigen::Vector3d P_ecef(_tmp34, _tmp38, _tmp4333);
    if (P_ecef.norm() > 0)
    {
        // 计算卫星的方位角/仰角
        sat_azel(P_ecef, sv_pos, azel);
        Eigen::Vector3d rcv_lla = ecef2geo(P_ecef);
        tro_delay = calculate_trop_delay(obs->time, rcv_lla, azel);
        ion_delay = calculate_ion_delay(obs->time, iono_paras, rcv_lla, azel);
    }
    double sin_el = sin(azel[1]);
    double sin_el_2 = sin_el*sin_el;
    double pr_weight = sin_el_2 / pr_uura * relative_sqrt_info;
    double dp_weight = sin_el_2 / dp_uura * relative_sqrt_info * PSR_TO_DOPP_RATIO;



    const Scalar _tmp43 =
        -_tmp1 * _tmp23 - _tmp22 * _tmp42 - _tmp30 * _tmp41 - ref_ecef(2, 0) + sv_pos(2, 0);
    const Scalar _tmp44 = std::sqrt(Scalar(std::pow(_tmp39, Scalar(2)) + std::pow(_tmp40, Scalar(2)) +
                                            std::pow(_tmp43, Scalar(2)) + epsilon));
    const Scalar _tmp45 = -Scalar(2.4323877909897198e-13) * _tmp34 * sv_pos(1, 0) +
                            Scalar(2.4323877909897198e-13) * _tmp38 * sv_pos(0, 0) + _tmp44 +
                            ion_delay - psr_measured + rcv_dt - Scalar(299792458.0) * svdt +
                            Scalar(299792458.0) * tgd + tro_delay;
    const Scalar _tmp46 = Vi(2, 0) * ratio + Vj(2, 0) * _tmp0;
    const Scalar _tmp47 = Vi(0, 0) * ratio + Vj(0, 0) * _tmp0;
    const Scalar _tmp48 = Vi(1, 0) * ratio + Vj(1, 0) * _tmp0;
    const Scalar _tmp49 = _tmp21 * _tmp46 + _tmp29 * _tmp47 + _tmp33 * _tmp48;
    const Scalar _tmp50 = _tmp35 * _tmp46 + _tmp36 * _tmp47 + _tmp37 * _tmp48;
    const Scalar _tmp51 = Scalar(1.0) / (_tmp44);
    const Scalar _tmp52 =
        -Scalar(2.4323877909897299e-13) * _tmp34 * sv_vel(1, 0) +
        Scalar(2.4323877909897299e-13) * _tmp38 * sv_vel(0, 0) +
        _tmp39 * _tmp51 * (-_tmp49 + sv_vel(0, 0)) + _tmp40 * _tmp51 * (-_tmp50 + sv_vel(1, 0)) +
        _tmp43 * _tmp51 * (-_tmp23 * _tmp46 - _tmp41 * _tmp48 - _tmp42 * _tmp47 + sv_vel(2, 0)) -
        Scalar(2.4323877909897299e-13) * _tmp49 * sv_pos(1, 0) +
        Scalar(2.4323877909897299e-13) * _tmp50 * sv_pos(0, 0) +
        Scalar(299792458.0) * dopp_measured / freq + rcv_ddt - Scalar(299792458.0) * svddt;
    const Scalar _tmp53 = Scalar(2.4323877909897198e-13) * sv_pos(0, 0);
    const Scalar _tmp54 = std::sin(_tmp18);
    const Scalar _tmp55 = _tmp3;
    const Scalar _tmp56 = std::sin(_tmp55);
    const Scalar _tmp57 = _tmp24 * _tmp56;
    const Scalar _tmp58 = _tmp54 * _tmp57;
    const Scalar _tmp59 = std::cos(_tmp55);
    const Scalar _tmp60 = _tmp27 * _tmp59;
    const Scalar _tmp61 = -_tmp58 + _tmp60;
    const Scalar _tmp62 = _tmp61 * ratio;
    const Scalar _tmp63 = Scalar(2.4323877909897198e-13) * sv_pos(1, 0);
    const Scalar _tmp64 = _tmp24 * _tmp59;
    const Scalar _tmp65 = _tmp54 * _tmp64;
    const Scalar _tmp66 = _tmp27 * _tmp56;
    const Scalar _tmp67 = -_tmp65 - _tmp66;
    const Scalar _tmp68 = _tmp67 * ratio;
    const Scalar _tmp69 = std::cos(_tmp18);
    const Scalar _tmp70 = _tmp24 * _tmp69;
    const Scalar _tmp71 = _tmp1 * _tmp54;
    const Scalar _tmp72 = _tmp27 * _tmp69;
    const Scalar _tmp73 = -_tmp22 * _tmp70 - _tmp30 * _tmp72 - _tmp71 - ref_ecef(2, 0) + sv_pos(2, 0);
    const Scalar _tmp74 = 2 * _tmp73;
    const Scalar _tmp75 = _tmp74 * ratio;
    const Scalar _tmp76 = _tmp54 * _tmp66;
    const Scalar _tmp77 = -_tmp64 - _tmp76;
    const Scalar _tmp78 = _tmp1 * _tmp69;
    const Scalar _tmp79 = _tmp56 * _tmp78;
    const Scalar _tmp80 = -_tmp22 * _tmp61 - _tmp30 * _tmp77 - _tmp79 - ref_ecef(1, 0) + sv_pos(1, 0);
    const Scalar _tmp81 = 2 * _tmp80;
    const Scalar _tmp82 = _tmp54 * _tmp60;
    const Scalar _tmp83 = _tmp57 - _tmp82;
    const Scalar _tmp84 = _tmp59 * _tmp78;
    const Scalar _tmp85 = -_tmp22 * _tmp67 - _tmp30 * _tmp83 - _tmp84 - ref_ecef(0, 0) + sv_pos(0, 0);
    const Scalar _tmp86 = 2 * _tmp85;
    const Scalar _tmp87 = -_tmp62 * _tmp81 - _tmp68 * _tmp86 - _tmp70 * _tmp75;
    const Scalar _tmp88 = std::pow(_tmp73, Scalar(2)) + std::pow(_tmp80, Scalar(2)) +
                            std::pow(_tmp85, Scalar(2)) + epsilon;
    const Scalar _tmp89 = std::pow(_tmp88, Scalar(Scalar(-1) / Scalar(2)));
    const Scalar _tmp90 = (Scalar(1) / Scalar(2)) * _tmp89;
    const Scalar _tmp91 = _tmp53 * _tmp62 - _tmp63 * _tmp68 + _tmp87 * _tmp90;
    const Scalar _tmp92 = (Scalar(1) / Scalar(2)) / (_tmp88 * std::sqrt(_tmp88));
    const Scalar _tmp93 = _tmp46 * _tmp69;
    const Scalar _tmp94 = _tmp56 * _tmp93;
    const Scalar _tmp95 = -_tmp47 * _tmp61 - _tmp48 * _tmp77 - _tmp94 + sv_vel(1, 0);
    const Scalar _tmp96 = _tmp80 * _tmp95;
    const Scalar _tmp97 = _tmp92 * _tmp96;
    const Scalar _tmp98 = _tmp59 * _tmp93;
    const Scalar _tmp99 = -_tmp47 * _tmp67 - _tmp48 * _tmp83 - _tmp98 + sv_vel(0, 0);
    const Scalar _tmp100 = _tmp85 * _tmp99;
    const Scalar _tmp101 = _tmp100 * _tmp92;
    const Scalar _tmp102 = _tmp89 * _tmp99;
    const Scalar _tmp103 = Scalar(2.4323877909897299e-13) * sv_vel(0, 0);
    const Scalar _tmp104 = Scalar(2.4323877909897299e-13) * sv_vel(1, 0);
    const Scalar _tmp105 = _tmp46 * _tmp54;
    const Scalar _tmp106 = -_tmp105 - _tmp47 * _tmp70 - _tmp48 * _tmp72 + sv_vel(2, 0);
    const Scalar _tmp107 = _tmp106 * _tmp89;
    const Scalar _tmp108 = _tmp107 * ratio;
    const Scalar _tmp109 = _tmp106 * _tmp73;
    const Scalar _tmp110 = _tmp109 * _tmp92;
    const Scalar _tmp111 = _tmp89 * _tmp95;
    const Scalar _tmp112 = _tmp111 * ratio;
    const Scalar _tmp113 = -_tmp101 * _tmp87 - _tmp102 * _tmp68 + _tmp103 * _tmp62 -
                            _tmp104 * _tmp68 - _tmp108 * _tmp70 - _tmp110 * _tmp87 - _tmp112 * _tmp61 -
                            _tmp87 * _tmp97;
    const Scalar _tmp114 = _tmp77 * ratio;
    const Scalar _tmp115 = _tmp83 * ratio;
    const Scalar _tmp116 = -_tmp114 * _tmp81 - _tmp115 * _tmp86 - _tmp72 * _tmp75;
    const Scalar _tmp117 = _tmp114 * _tmp53 - _tmp115 * _tmp63 + _tmp116 * _tmp90;
    const Scalar _tmp118 = -_tmp101 * _tmp116 - _tmp102 * _tmp115 + _tmp103 * _tmp114 -
                            _tmp104 * _tmp115 - _tmp108 * _tmp72 - _tmp110 * _tmp116 -
                            _tmp112 * _tmp77 - _tmp116 * _tmp97;
    const Scalar _tmp119 = _tmp56 * _tmp69;
    const Scalar _tmp120 = _tmp119 * ratio;
    const Scalar _tmp121 = _tmp59 * _tmp69;
    const Scalar _tmp122 = _tmp121 * ratio;
    const Scalar _tmp123 = _tmp54 * ratio;
    const Scalar _tmp124 = -_tmp120 * _tmp81 - _tmp122 * _tmp86 - _tmp123 * _tmp74;
    const Scalar _tmp125 = _tmp120 * _tmp53 - _tmp122 * _tmp63 + _tmp124 * _tmp90;
    const Scalar _tmp126 = _tmp124 * _tmp92;
    const Scalar _tmp127 = -_tmp101 * _tmp124 - _tmp102 * _tmp122 + _tmp103 * _tmp120 -
                            _tmp104 * _tmp122 - _tmp107 * _tmp123 - _tmp109 * _tmp126 -
                            _tmp112 * _tmp119 - _tmp126 * _tmp96;
    const Scalar _tmp128 = _tmp80 * _tmp89;
    const Scalar _tmp129 = _tmp85 * _tmp89;
    const Scalar _tmp130 = _tmp73 * _tmp89;
    const Scalar _tmp131 = _tmp130 * ratio;
    const Scalar _tmp132 = Scalar(2.4323877909897299e-13) * sv_pos(0, 0);
    const Scalar _tmp133 = Scalar(2.4323877909897299e-13) * sv_pos(1, 0);
    const Scalar _tmp134 =
        -_tmp128 * _tmp62 - _tmp129 * _tmp68 - _tmp131 * _tmp70 + _tmp132 * _tmp62 - _tmp133 * _tmp68;
    const Scalar _tmp135 = -_tmp114 * _tmp128 + _tmp114 * _tmp132 - _tmp115 * _tmp129 -
                            _tmp115 * _tmp133 - _tmp131 * _tmp72;
    const Scalar _tmp136 = -_tmp120 * _tmp128 + _tmp120 * _tmp132 - _tmp122 * _tmp129 -
                            _tmp122 * _tmp133 - _tmp123 * _tmp130;
    const Scalar _tmp137 = _tmp0 * _tmp63;
    const Scalar _tmp138 = _tmp0 * _tmp61;
    const Scalar _tmp139 = _tmp0 * _tmp74;
    const Scalar _tmp140 = _tmp0 * _tmp86;
    const Scalar _tmp141 = -_tmp138 * _tmp81 - _tmp139 * _tmp70 - _tmp140 * _tmp67;
    const Scalar _tmp142 = -_tmp137 * _tmp67 + _tmp138 * _tmp53 + _tmp141 * _tmp90;
    const Scalar _tmp143 = _tmp141 * _tmp92;
    const Scalar _tmp144 = _tmp0 * _tmp102;
    const Scalar _tmp145 = _tmp0 * _tmp104;
    const Scalar _tmp146 = _tmp0 * _tmp107;
    const Scalar _tmp147 = -_tmp101 * _tmp141 + _tmp103 * _tmp138 - _tmp109 * _tmp143 -
                            _tmp111 * _tmp138 - _tmp143 * _tmp96 - _tmp144 * _tmp67 -
                            _tmp145 * _tmp67 - _tmp146 * _tmp70;
    const Scalar _tmp148 = _tmp0 * _tmp77;
    const Scalar _tmp149 = -_tmp139 * _tmp72 - _tmp140 * _tmp83 - _tmp148 * _tmp81;
    const Scalar _tmp150 = -_tmp137 * _tmp83 + _tmp148 * _tmp53 + _tmp149 * _tmp90;
    const Scalar _tmp151 = -_tmp101 * _tmp149 + _tmp103 * _tmp148 - _tmp110 * _tmp149 -
                            _tmp111 * _tmp148 - _tmp144 * _tmp83 - _tmp145 * _tmp83 -
                            _tmp146 * _tmp72 - _tmp149 * _tmp97;
    const Scalar _tmp152 = _tmp0 * _tmp119;
    const Scalar _tmp153 = _tmp0 * _tmp54;
    const Scalar _tmp154 = -_tmp121 * _tmp140 - _tmp152 * _tmp81 - _tmp153 * _tmp74;
    const Scalar _tmp155 = -_tmp121 * _tmp137 + _tmp152 * _tmp53 + _tmp154 * _tmp90;
    const Scalar _tmp156 = -_tmp101 * _tmp154 + _tmp103 * _tmp152 - _tmp107 * _tmp153 -
                            _tmp110 * _tmp154 - _tmp111 * _tmp152 - _tmp121 * _tmp144 -
                            _tmp121 * _tmp145 - _tmp154 * _tmp97;
    const Scalar _tmp157 = _tmp0 * _tmp129;
    const Scalar _tmp158 = _tmp0 * _tmp130;
    const Scalar _tmp159 = _tmp0 * _tmp133;
    const Scalar _tmp160 = -_tmp128 * _tmp138 + _tmp132 * _tmp138 - _tmp157 * _tmp67 -
                            _tmp158 * _tmp70 - _tmp159 * _tmp67;
    const Scalar _tmp161 = -_tmp128 * _tmp148 + _tmp132 * _tmp148 - _tmp157 * _tmp83 -
                            _tmp158 * _tmp72 - _tmp159 * _tmp83;
    const Scalar _tmp162 = -_tmp121 * _tmp157 - _tmp121 * _tmp159 - _tmp128 * _tmp152 -
                            _tmp130 * _tmp153 + _tmp132 * _tmp152;
    const Scalar _tmp163 = _tmp58 - _tmp60;
    const Scalar _tmp164 = _tmp163 * _tmp30;
    const Scalar _tmp165 = _tmp22 * _tmp77;
    const Scalar _tmp166 = _tmp164 + _tmp165;
    const Scalar _tmp167 = -_tmp22 * _tmp72 + _tmp30 * _tmp70;
    const Scalar _tmp168 = -_tmp164 - _tmp165;
    const Scalar _tmp169 = _tmp65 + _tmp66;
    const Scalar _tmp170 = _tmp169 * _tmp30;
    const Scalar _tmp171 = _tmp22 * _tmp83;
    const Scalar _tmp172 = -_tmp170 - _tmp171;
    const Scalar _tmp173 = _tmp167 * _tmp74 + _tmp168 * _tmp81 + _tmp172 * _tmp86;
    const Scalar _tmp174 = _tmp170 + _tmp171;
    const Scalar _tmp175 = _tmp166 * _tmp53 + _tmp173 * _tmp90 - _tmp174 * _tmp63;
    const Scalar _tmp176 = _tmp173 * _tmp92;
    const Scalar _tmp177 = _tmp47 * _tmp83;
    const Scalar _tmp178 = _tmp169 * _tmp48;
    const Scalar _tmp179 = _tmp47 * _tmp77;
    const Scalar _tmp180 = _tmp163 * _tmp48;
    const Scalar _tmp181 =
        -_tmp101 * _tmp173 + _tmp102 * _tmp172 + _tmp103 * _tmp166 - _tmp104 * _tmp174 +
        _tmp107 * _tmp167 - _tmp109 * _tmp176 + _tmp111 * _tmp168 + _tmp128 * (-_tmp179 - _tmp180) +
        _tmp129 * (-_tmp177 - _tmp178) + _tmp130 * (-_tmp47 * _tmp72 + _tmp48 * _tmp70) +
        _tmp132 * (_tmp179 + _tmp180) - _tmp133 * (_tmp177 + _tmp178) - _tmp176 * _tmp96;
    const Scalar _tmp182 = Scalar(1.0) / (_tmp15);
    const Scalar _tmp183 = std::pow(_tmp9, Scalar(Scalar(-5) / Scalar(2)));
    const Scalar _tmp184 = _tmp14 * _tmp183;
    const Scalar _tmp185 = Scalar(1.3295415302198599e+39) * _tmp184;
    const Scalar _tmp186 = Scalar(3.2902684119959e+25) * _tmp10 * _tmp15;
    const Scalar _tmp187 = std::pow(_tmp16, Scalar(-2));
    const Scalar _tmp188 = _tmp13 * _tmp187;
    const Scalar _tmp189 = _tmp11 * _tmp183;
    const Scalar _tmp190 = Scalar(5.44511055670503e+52) * _tmp17 * _tmp189 * _tmp5;
    const Scalar _tmp191 = Scalar(1.0) / (std::pow(_tmp13, Scalar(2)) * _tmp187 + 1);
    const Scalar _tmp192 =
        _tmp191 *
        (-_tmp188 * (_tmp182 * ref_ecef(0, 0) + _tmp185 * ref_ecef(0, 0) - _tmp186 * ref_ecef(0, 0)) -
        _tmp190 * ref_ecef(0, 0));
    const Scalar _tmp193 = _tmp56 * _tmp71;
    const Scalar _tmp194 = _tmp192 * _tmp193;
    const Scalar _tmp195 = Scalar(1.0) / (std::pow(_tmp2, Scalar(2)) + _tmp7);
    const Scalar _tmp196 = _tmp195 * ref_ecef(1, 0);
    const Scalar _tmp197 = _tmp192 * _tmp69;
    const Scalar _tmp198 = -_tmp196 * _tmp57 + _tmp196 * _tmp82 - _tmp197 * _tmp66;
    const Scalar _tmp199 = _tmp198 * _tmp30;
    const Scalar _tmp200 = _tmp196 * _tmp65 + _tmp196 * _tmp66 - _tmp197 * _tmp57;
    const Scalar _tmp201 = _tmp200 * _tmp22;
    const Scalar _tmp202 = _tmp196 * _tmp84;
    const Scalar _tmp203 = _tmp194 - _tmp199 - _tmp201 + _tmp202;
    const Scalar _tmp204 = _tmp24 * _tmp54;
    const Scalar _tmp205 = _tmp192 * _tmp204;
    const Scalar _tmp206 = _tmp27 * _tmp54;
    const Scalar _tmp207 = _tmp192 * _tmp206;
    const Scalar _tmp208 = -_tmp192 * _tmp78 + _tmp205 * _tmp22 + _tmp207 * _tmp30;
    const Scalar _tmp209 = -_tmp196 * _tmp64 - _tmp196 * _tmp76 - _tmp197 * _tmp60;
    const Scalar _tmp210 = _tmp209 * _tmp30;
    const Scalar _tmp211 = -_tmp196 * _tmp58 + _tmp196 * _tmp60 - _tmp197 * _tmp64;
    const Scalar _tmp212 = _tmp211 * _tmp22;
    const Scalar _tmp213 = _tmp196 * _tmp79;
    const Scalar _tmp214 = _tmp59 * _tmp71;
    const Scalar _tmp215 = _tmp192 * _tmp214;
    const Scalar _tmp216 = -_tmp210 - _tmp212 - _tmp213 + _tmp215 - 1;
    const Scalar _tmp217 = _tmp203 * _tmp81 + _tmp208 * _tmp74 + _tmp216 * _tmp86;
    const Scalar _tmp218 = _tmp210 + _tmp212 + _tmp213 - _tmp215 + 1;
    const Scalar _tmp219 = -_tmp194 + _tmp199 + _tmp201 - _tmp202;
    const Scalar _tmp220 = _tmp217 * _tmp90 - _tmp218 * _tmp63 + _tmp219 * _tmp53;
    const Scalar _tmp221 = _tmp211 * _tmp47;
    const Scalar _tmp222 = _tmp209 * _tmp48;
    const Scalar _tmp223 = _tmp105 * _tmp59;
    const Scalar _tmp224 = _tmp192 * _tmp223;
    const Scalar _tmp225 = _tmp196 * _tmp94;
    const Scalar _tmp226 = _tmp217 * _tmp92;
    const Scalar _tmp227 = _tmp105 * _tmp56;
    const Scalar _tmp228 = _tmp192 * _tmp227;
    const Scalar _tmp229 = _tmp200 * _tmp47;
    const Scalar _tmp230 = _tmp198 * _tmp48;
    const Scalar _tmp231 = _tmp196 * _tmp98;
    const Scalar _tmp232 = -_tmp100 * _tmp226 + _tmp102 * _tmp216 + _tmp103 * _tmp219 -
                            _tmp104 * _tmp218 + _tmp107 * _tmp208 - _tmp109 * _tmp226 +
                            _tmp111 * _tmp203 + _tmp128 * (_tmp228 - _tmp229 - _tmp230 + _tmp231) +
                            _tmp129 * (-_tmp221 - _tmp222 + _tmp224 - _tmp225) +
                            _tmp130 * (-_tmp192 * _tmp93 + _tmp205 * _tmp47 + _tmp207 * _tmp48) +
                            _tmp132 * (-_tmp228 + _tmp229 + _tmp230 - _tmp231) -
                            _tmp133 * (_tmp221 + _tmp222 - _tmp224 + _tmp225) - _tmp226 * _tmp96;
    const Scalar _tmp233 = _tmp195 * _tmp2;
    const Scalar _tmp234 = _tmp233 * _tmp84;
    const Scalar _tmp235 =
        _tmp191 *
        (-_tmp188 * (_tmp182 * ref_ecef(1, 0) + _tmp185 * ref_ecef(1, 0) - _tmp186 * ref_ecef(1, 0)) -
        _tmp190 * ref_ecef(1, 0));
    const Scalar _tmp236 = _tmp235 * _tmp69;
    const Scalar _tmp237 = -_tmp233 * _tmp65 - _tmp233 * _tmp66 - _tmp236 * _tmp57;
    const Scalar _tmp238 = _tmp22 * _tmp237;
    const Scalar _tmp239 = _tmp233 * _tmp57 - _tmp233 * _tmp82 - _tmp236 * _tmp66;
    const Scalar _tmp240 = _tmp239 * _tmp30;
    const Scalar _tmp241 = _tmp193 * _tmp235;
    const Scalar _tmp242 = -_tmp234 - _tmp238 - _tmp240 + _tmp241 - 1;
    const Scalar _tmp243 = _tmp204 * _tmp22;
    const Scalar _tmp244 = _tmp206 * _tmp30;
    const Scalar _tmp245 = _tmp235 * _tmp243 + _tmp235 * _tmp244 - _tmp235 * _tmp78;
    const Scalar _tmp246 = _tmp214 * _tmp235;
    const Scalar _tmp247 = _tmp233 * _tmp79;
    const Scalar _tmp248 = _tmp233 * _tmp58 - _tmp233 * _tmp60 - _tmp236 * _tmp64;
    const Scalar _tmp249 = _tmp22 * _tmp248;
    const Scalar _tmp250 = _tmp233 * _tmp64 + _tmp233 * _tmp76 - _tmp236 * _tmp60;
    const Scalar _tmp251 = _tmp250 * _tmp30;
    const Scalar _tmp252 = _tmp246 + _tmp247 - _tmp249 - _tmp251;
    const Scalar _tmp253 = _tmp242 * _tmp81 + _tmp245 * _tmp74 + _tmp252 * _tmp86;
    const Scalar _tmp254 = _tmp234 + _tmp238 + _tmp240 - _tmp241 + 1;
    const Scalar _tmp255 = -_tmp246 - _tmp247 + _tmp249 + _tmp251;
    const Scalar _tmp256 = _tmp253 * _tmp90 + _tmp254 * _tmp53 - _tmp255 * _tmp63;
    const Scalar _tmp257 = _tmp250 * _tmp48;
    const Scalar _tmp258 = _tmp248 * _tmp47;
    const Scalar _tmp259 = _tmp233 * _tmp94;
    const Scalar _tmp260 = _tmp223 * _tmp235;
    const Scalar _tmp261 = _tmp233 * _tmp98;
    const Scalar _tmp262 = _tmp227 * _tmp235;
    const Scalar _tmp263 = _tmp239 * _tmp48;
    const Scalar _tmp264 = _tmp237 * _tmp47;
    const Scalar _tmp265 = _tmp204 * _tmp47;
    const Scalar _tmp266 =
        -_tmp101 * _tmp253 + _tmp102 * _tmp252 + _tmp103 * _tmp254 - _tmp104 * _tmp255 +
        _tmp107 * _tmp245 - _tmp110 * _tmp253 + _tmp111 * _tmp242 +
        _tmp128 * (-_tmp261 + _tmp262 - _tmp263 - _tmp264) +
        _tmp129 * (-_tmp257 - _tmp258 + _tmp259 + _tmp260) +
        _tmp130 * (_tmp206 * _tmp235 * _tmp48 + _tmp235 * _tmp265 - _tmp235 * _tmp93) +
        _tmp132 * (_tmp261 - _tmp262 + _tmp263 + _tmp264) -
        _tmp133 * (_tmp257 + _tmp258 - _tmp259 - _tmp260) - _tmp253 * _tmp97;
    const Scalar _tmp267 =
        _tmp191 *
        (_tmp17 * (Scalar(1.34752280070479e+39) * _tmp12 * _tmp6 -
                    Scalar(5.4818078615632705e+52) * _tmp189 * std::pow(ref_ecef(2, 0), Scalar(4)) +
                    Scalar(1.0)) -
        Scalar(1.3385019710315001e+39) * _tmp184 * _tmp188 * ref_ecef(2, 0));
    const Scalar _tmp268 = _tmp267 * _tmp69;
    const Scalar _tmp269 = _tmp22 * _tmp268;
    const Scalar _tmp270 = _tmp269 * _tmp64;
    const Scalar _tmp271 = _tmp268 * _tmp30;
    const Scalar _tmp272 = _tmp271 * _tmp60;
    const Scalar _tmp273 = _tmp214 * _tmp267;
    const Scalar _tmp274 = -_tmp270 - _tmp272 - _tmp273;
    const Scalar _tmp275 = _tmp193 * _tmp267;
    const Scalar _tmp276 = _tmp271 * _tmp66;
    const Scalar _tmp277 = _tmp269 * _tmp57;
    const Scalar _tmp278 = -_tmp275 - _tmp276 - _tmp277;
    const Scalar _tmp279 = _tmp270 + _tmp272 + _tmp273;
    const Scalar _tmp280 = _tmp243 * _tmp267 + _tmp244 * _tmp267 - _tmp267 * _tmp78 - 1;
    const Scalar _tmp281 = _tmp275 + _tmp276 + _tmp277;
    const Scalar _tmp282 = _tmp279 * _tmp86 + _tmp280 * _tmp74 + _tmp281 * _tmp81;
    const Scalar _tmp283 = -_tmp274 * _tmp63 + _tmp278 * _tmp53 + _tmp282 * _tmp90;
    const Scalar _tmp284 = _tmp223 * _tmp267;
    const Scalar _tmp285 = _tmp268 * _tmp47;
    const Scalar _tmp286 = _tmp285 * _tmp64;
    const Scalar _tmp287 = _tmp267 * _tmp48;
    const Scalar _tmp288 = _tmp287 * _tmp69;
    const Scalar _tmp289 = _tmp288 * _tmp60;
    const Scalar _tmp290 = _tmp288 * _tmp66;
    const Scalar _tmp291 = _tmp285 * _tmp57;
    const Scalar _tmp292 = _tmp227 * _tmp267;
    const Scalar _tmp293 = -_tmp101 * _tmp282 + _tmp102 * _tmp279 + _tmp103 * _tmp278 -
                            _tmp104 * _tmp274 + _tmp107 * _tmp280 - _tmp110 * _tmp282 +
                            _tmp111 * _tmp281 + _tmp128 * (_tmp290 + _tmp291 + _tmp292) +
                            _tmp129 * (_tmp284 + _tmp286 + _tmp289) +
                            _tmp130 * (_tmp206 * _tmp287 + _tmp265 * _tmp267 - _tmp267 * _tmp93) +
                            _tmp132 * (-_tmp290 - _tmp291 - _tmp292) -
                            _tmp133 * (-_tmp284 - _tmp286 - _tmp289) - _tmp282 * _tmp97;
    /*const Scalar _tmp294 = std::pow(dp_weight, Scalar(2));
    const Scalar _tmp295 = std::pow(pr_weight, Scalar(2));
    const Scalar _tmp296 = _tmp118 * _tmp294;
    const Scalar _tmp297 = _tmp117 * _tmp295;
    const Scalar _tmp298 = _tmp125 * _tmp295;
    const Scalar _tmp299 = _tmp113 * _tmp294;
    const Scalar _tmp300 = _tmp134 * _tmp294;
    const Scalar _tmp301 = _tmp135 * _tmp294;
    const Scalar _tmp302 = _tmp136 * _tmp294;
    const Scalar _tmp303 = _tmp295 * _tmp91;
    const Scalar _tmp304 = _tmp147 * _tmp294;
    const Scalar _tmp305 = _tmp151 * _tmp294;
    const Scalar _tmp306 = _tmp150 * _tmp295;
    const Scalar _tmp307 = _tmp155 * _tmp295;
    const Scalar _tmp308 = _tmp156 * _tmp294;
    const Scalar _tmp309 = _tmp160 * _tmp294;
    const Scalar _tmp310 = _tmp161 * _tmp294;
    const Scalar _tmp311 = _tmp162 * _tmp294;
    const Scalar _tmp312 = _tmp181 * _tmp294;
    const Scalar _tmp313 = _tmp232 * _tmp294;
    const Scalar _tmp314 = _tmp220 * _tmp295;
    const Scalar _tmp315 = _tmp266 * _tmp294;
    const Scalar _tmp316 = _tmp293 * _tmp294;
    const Scalar _tmp317 = _tmp142 * _tmp295;
    const Scalar _tmp318 = _tmp175 * _tmp295;
    const Scalar _tmp319 = _tmp256 * _tmp295;
    const Scalar _tmp320 = _tmp294 * _tmp52;
    const Scalar _tmp321 = _tmp295 * _tmp45;
    */

    // Output terms (4)
    // if (res != nullptr) {
        // Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

        residuals[0] = _tmp45 * pr_weight;
        residuals[1] = _tmp52 * dp_weight;
    // }

    // if (jacobian != nullptr) 
    if (jacobians) 
    {
        // Eigen::Matrix<Scalar, 2, 18>& _jacobian = (*jacobian);

        // J_Pi
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_Pi(jacobians[0]);
            J_Pi.setZero();

            J_Pi(0, 0) = _tmp91 * pr_weight;
            J_Pi(1, 0) = _tmp113 * dp_weight;
            J_Pi(0, 1) = _tmp117 * pr_weight;
            J_Pi(1, 1) = _tmp118 * dp_weight;
            J_Pi(0, 2) = _tmp125 * pr_weight;
            J_Pi(1, 2) = _tmp127 * dp_weight;
        }

        // J_Vi
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_Vi(jacobians[1]);
            J_Vi.setZero();
            J_Vi(0, 0) = 0;
            J_Vi(1, 0) = _tmp134 * dp_weight;
            J_Vi(0, 1) = 0;
            J_Vi(1, 1) = _tmp135 * dp_weight;
            J_Vi(0, 2) = 0;
            J_Vi(1, 2) = _tmp136 * dp_weight;
        }

        // J_Pj
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_Pj(jacobians[2]);
            J_Pj.setZero();

            J_Pj(0, 0) = _tmp142 * pr_weight;
            J_Pj(1, 0) = _tmp147 * dp_weight;
            J_Pj(0, 1) = _tmp150 * pr_weight;
            J_Pj(1, 1) = _tmp151 * dp_weight;
            J_Pj(0, 2) = _tmp155 * pr_weight;
            J_Pj(1, 2) = _tmp156 * dp_weight;
        }

        // J_Vj
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_Vj(jacobians[3]);
            J_Vj.setZero();

            J_Vj(0, 0) = 0;
            J_Vj(1, 0) = _tmp160 * dp_weight;
            J_Vj(0, 1) = 0;
            J_Vj(1, 1) = _tmp161 * dp_weight;
            J_Vj(0, 2) = 0;
            J_Vj(1, 2) = _tmp162 * dp_weight;
        }
        
        // J_rcv_dt
        if (jacobians[4])
        {
            jacobians[4][0] = pr_weight;
            jacobians[4][1] = 0;
        }

        // J_rcv_ddt
        if (jacobians[5])
        {
            jacobians[5][0] = 0;
            jacobians[5][1] = dp_weight;
        }

        // J_yaw_diff
        if (jacobians[6])
        {
            jacobians[6][0] = _tmp175 * pr_weight;
            jacobians[6][1] = _tmp181 * dp_weight;
        }

        // J_ref_ecef, approximation for simplicity
        if (jacobians[7])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_ref_ecef(jacobians[7]);
            J_ref_ecef.setZero();

            J_ref_ecef(0, 0) = _tmp220 * pr_weight;
            J_ref_ecef(1, 0) = _tmp232 * dp_weight;
            J_ref_ecef(0, 1) = _tmp256 * pr_weight;
            J_ref_ecef(1, 1) = _tmp266 * dp_weight;
            J_ref_ecef(0, 2) = _tmp283 * pr_weight;
            J_ref_ecef(1, 2) = _tmp293 * dp_weight;
        }
    }
    ///////////////////////////////////////////////////////////////
    // auto end = std::chrono::steady_clock::now(); // 2024-7-2
    // auto diff = end - start;
    // std::cout << "sym::gnss_psr_dopp_factor: " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    return true;
}