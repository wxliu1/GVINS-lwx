import symforce
symforce.set_epsilon_to_symbol()

from symforce import typing as T # 导入symforce包里面的typing模块到当前模块的命名空间，并重命名为T

from symforce import codegen

import symforce.symbolic as sf
from symforce.notebook_util import display

import shutil
from pathlib import Path

M_PI = 3.14159265358979323846
LIGHT_SPEED = 2.99792458e8
EARTH_ECCE_2 = 6.69437999014e-3
EARTH_SEMI_MAJOR = 6378137
R2D = 180.0 / M_PI
D2R = M_PI / 180.0
EARTH_OMG_GPS = 7.2921151467e-5
relative_sqrt_info = 10.0
PSR_TO_DOPP_RATIO = 5


# ecef2rotation：根据anchor point的坐标计算ENU系到ECEF系的旋转
from symengine import atan
# def ecef2rotation(xyz: sf.V3, epsilon: sf.Scalar) -> sf.Matrix33:
def ecef2geo(xyz: sf.V3, epsilon: sf.Scalar = 0.0) -> sf.V3:
    # step 1: ecef2geo 计算得到纬度，经度，（海拔）高度
    lla = sf.V3.zero()
    # TODO: LLA coordinate is not defined if x = 0 and y = 0
    # if xyz.x == 0 && xyz.y == 0 : return lla
    # 可是函数不允许有分支, 这里引入epsilon视图解决问题

    e2 = EARTH_ECCE_2
    a = EARTH_SEMI_MAJOR
    a2 = a * a
    b2 = a2 * (1 - e2)
    b = sf.sqrt(b2)
    ep2 = (a2 - b2) / (b2 + epsilon)
    x, y, z = xyz
    xy = sf.V2(x, y)
    p = xy.norm(epsilon=epsilon)

    # two sides and hypotenuse of right angle triangle with one angle = theta:
    s1 = xyz.z * a
    s2 = p * b
    h = sf.sqrt(s1 * s1 + s2 * s2 + epsilon**2)
    sin_theta = s1 / h
    cos_theta = s2 / h

    # two sides and hypotenuse of right angle triangle with one angle = lat:
    # s1 = xyz.z + ep2 * b * (sin_theta ** 3)
    s1 = xyz.z + ep2 * b * sf.Pow(sin_theta, 3)
    s2 = p - a * e2 * sf.Pow(cos_theta, 3)
    h = sf.sqrt(s1 * s1 + s2 * s2 + epsilon**2)
    tan_lat = s1 / (s2 + epsilon)
    sin_lat = s1 / (h + epsilon)
    cos_lat = s2 / (h + epsilon)
    lat = atan(tan_lat)
    lat_deg = lat * R2D

    N = a2 * sf.Pow((a2 * cos_lat * cos_lat + b2 * sin_lat * sin_lat), -0.5)
    altM = p / (cos_lat + epsilon) - N

    lon = sf.atan2(xyz.y, xyz.x, epsilon=epsilon)
    lon_deg = lon * R2D
    lla = sf.V3(lat_deg, lon_deg, altM)
    return lla

def ecef2rotation(xyz: sf.V3, epsilon: sf.Scalar) -> sf.Matrix33:
    # step2: geo2rotation 计算ENU系到ECEF系的旋转
    # ref_geo = lla
    ref_geo = ecef2geo(xyz, epsilon = epsilon)
    lat = ref_geo.x * D2R
    lon = ref_geo.y * D2R
    sin_lat = sf.sin(lat)
    cos_lat = sf.cos(lat)
    sin_lon = sf.sin(lon)
    cos_lon = sf.cos(lon)

    # Given the ECEF coordinate of the anchor point, the rotation from ENU frame to ECEF frame is:
    R_ecef_enu = sf.Matrix33(-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon, \
                            cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon, \
                            0,  cos_lat, sin_lat)
    return R_ecef_enu

def ecef2enu(ref_lla: sf.V3, v_ecef: sf.V3) -> sf.V3:
    lat = ref_lla.x * D2R
    lon = ref_lla.y * D2R
    sin_lat = sf.sin(lat)
    cos_lat = sf.cos(lat)
    sin_lon = sf.sin(lon)
    cos_lon = sf.cos(lon)

    R_enu_ecef = sf.Matrix33(-sin_lon,             cos_lon,         0, \
                             -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat, \
                             cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat)
    return (R_enu_ecef * v_ecef)

# 计算卫星到接收机的方位角/仰角
def sat_azel(rev_pos: sf.V3, sat_pos: sf.V3, epsilon: sf.Scalar = 0) -> sf.V2:
    rev_lla = ecef2geo(rev_pos)
    rev2sat_ecef = (sat_pos - rev_pos).normalized() # .normalized(epsilon=epsilon) # .normalized(epsilon=0)
    rev2sat_enu = ecef2enu(rev_lla, rev2sat_ecef)
    
    # azel[0] = rev2sat_ecef.head<2>().norm() < 1e-12 ? 0.0 : atan2(rev2sat_enu.x(), rev2sat_enu.y());
    # azel[0] += (azel[0] < 0 ? 2*M_PI : 0);
    # azel[1] = asin(rev2sat_enu.z());

    # branchless singularity handling
    # TODO: 计算出的方位角，有可能为负吗？为负应该加上2PI，可是又不允许分支
    azimuth = sf.atan2(rev2sat_enu.x, rev2sat_enu.y, epsilon = epsilon)

    elevation = sf.asin(rev2sat_enu.z)

    return sf.V2(azimuth, elevation)


# gnss psr dopp residual: 2维
def gnss_psr_dopp_residual(
    # states:
    Pi: sf.V3,
    Vi: sf.V3,
    Pj: sf.V3,
    Vj: sf.V3,
    rcv_dt: sf.Scalar,
    rcv_ddt: sf.Scalar,
    yaw_diff: sf.Scalar,
    ref_ecef: sf.V3,
    # ?
    ion_delay: sf.Scalar, # uncomment it
    tro_delay: sf.Scalar, # uncomment it

    # precomputed:
    ratio: sf.Scalar,
    tgd: sf.Scalar,
    sv_pos: sf.V3,
    sv_vel: sf.V3,
    svdt: sf.Scalar,
    svddt: sf.Scalar,
    freq: sf.Scalar,
    psr_measured: sf.Scalar, # observation data pseudorange (m)
    dopp_measured: sf.Scalar, # observation data doppler frequency (Hz)
    pr_uura: sf.Scalar,
    dp_uura: sf.Scalar,
    pr_weight: sf.Scalar, # newly add
    dp_weight: sf.Scalar, # newly add
    epsilon: sf.Scalar = 0
) -> sf.V2:
    # construct residuals here.
    local_pos = ratio * Pi + (1.0 - ratio) * Pj
    local_vel = ratio * Vi + (1.0 - ratio) * Vj
    sin_yaw_diff = sf.sin(yaw_diff)
    cos_yaw_diff = sf.cos(yaw_diff)
    R_enu_local = sf.Matrix33(cos_yaw_diff, -sin_yaw_diff, 0, \
                              sin_yaw_diff, cos_yaw_diff, 0, \
                              0, 0, 1)
    # 计算地心地固坐标系下的位置和速度
    R_ecef_enu = ecef2rotation(ref_ecef, epsilon)
    R_ecef_local = R_ecef_enu * R_enu_local
    P_ecef = R_ecef_local * local_pos + ref_ecef
    V_ecef = R_ecef_local * local_vel

    # tmp comment
    # 计算卫星的方位角/仰角
    # azel = sat_azel(P_ecef, sv_pos, epsilon)
    # rcv_lla = ecef2geo(P_ecef)
    # TODO: 
    # tro_delay = calculate_trop_delay(obs->time, rcv_lla, azel)
    # ion_delay = calculate_ion_delay(obs->time, iono_paras, rcv_lla, azel)

    # sin_el = sf.sin(azel.y)
    # sin_el_2 = sin_el*sin_el
    # pr_weight = sin_el_2 / pr_uura * relative_sqrt_info
    # dp_weight = sin_el_2 / dp_uura * relative_sqrt_info * PSR_TO_DOPP_RATIO
    # the end.

    rcv2sat_ecef = sv_pos - P_ecef
    rcv2sat_unit = rcv2sat_ecef.normalized()

    psr_sagnac = EARTH_OMG_GPS*(sv_pos.x*P_ecef.y-sv_pos.y*P_ecef.x)/LIGHT_SPEED

    psr_estimated = rcv2sat_ecef.norm() + psr_sagnac + rcv_dt - svdt*LIGHT_SPEED + \
                    ion_delay + tro_delay + tgd*LIGHT_SPEED
    # psr_measured = obs->psr[freq_idx]
    r_pseudorange = (psr_estimated - psr_measured) * pr_weight

    dopp_sagnac = EARTH_OMG_GPS/LIGHT_SPEED*(sv_vel.x*P_ecef.y+ sv_pos.x*V_ecef.y - sv_vel.y*P_ecef.x - sv_pos.y*V_ecef.x)
    dopp_estimated = (sv_vel - V_ecef).dot(rcv2sat_unit) + dopp_sagnac + rcv_ddt - svddt*LIGHT_SPEED
    wavelength = LIGHT_SPEED / freq
    # dopp_measured = obs->dopp[freq_idx]
    r_doppler = (dopp_estimated + dopp_measured * wavelength) * dp_weight


    return sf.V2(r_pseudorange, r_doppler)



 # for gnss
def generate_gnss_residual_code(
    output_dir: T.Optional[Path] = None, print_code: bool = False
) -> None:
    gnss_codegen = codegen.Codegen.function(
        func=gnss_psr_dopp_residual,
        config=codegen.CppConfig(),
    )

    gnss_data = gnss_codegen.generate_function(output_dir)

    gnss_codegen_with_linearization = gnss_codegen.with_linearization(which_args=["Pi", "Vi", "Pj", "Vj", "rcv_dt", "rcv_ddt", "yaw_diff", "ref_ecef"])

    # 生成构建因子图的函数
    # Generate the function and print the code
    metadata = gnss_codegen_with_linearization.generate_function(
        output_dir=output_dir, skip_directory_nesting=False
    )


output_dir="/root/dev/python_ws/test_sym"

generate_gnss_residual_code(output_dir)


# for test

def calc_p_ecef(
    # states:
    Pi: sf.V3,
    Vi: sf.V3,
    Pj: sf.V3,
    Vj: sf.V3,
    rcv_dt: sf.Scalar,
    rcv_ddt: sf.Scalar,
    yaw_diff: sf.Scalar,
    ref_ecef: sf.V3,
    # ?
    # ion_delay: sf.Scalar, # uncomment it
    # tro_delay: sf.Scalar, # uncomment it

    # precomputed:
    ratio: sf.Scalar,
    # tgd: sf.Scalar,
    # sv_pos: sf.V3,
    # sv_vel: sf.V3,
    # svdt: sf.Scalar,
    # svddt: sf.Scalar,
    # freq: sf.Scalar,
    # psr_measured: sf.Scalar, # observation data pseudorange (m)
    # dopp_measured: sf.Scalar, # observation data doppler frequency (Hz)
    # pr_uura: sf.Scalar,
    # dp_uura: sf.Scalar,
    # pr_weight: sf.Scalar, # newly add
    # dp_weight: sf.Scalar, # newly add
    epsilon: sf.Scalar = 0
) -> sf.V3:
    # construct residuals here.
    local_pos = ratio * Pi + (1.0 - ratio) * Pj
    local_vel = ratio * Vi + (1.0 - ratio) * Vj
    sin_yaw_diff = sf.sin(yaw_diff)
    cos_yaw_diff = sf.cos(yaw_diff)
    R_enu_local = sf.Matrix33(cos_yaw_diff, -sin_yaw_diff, 0, \
                              sin_yaw_diff, cos_yaw_diff, 0, \
                              0, 0, 1)
    # 计算地心地固坐标系下的位置和速度
    R_ecef_enu = ecef2rotation(ref_ecef, epsilon)
    R_ecef_local = R_ecef_enu * R_enu_local
    P_ecef = R_ecef_local * local_pos + ref_ecef
    return P_ecef


calc_p_ecef_codegen = codegen.Codegen.function(
    func=calc_p_ecef,
    config=codegen.CppConfig(),
)
# calc_p_ecef_data = calc_p_ecef_codegen.generate_function(output_dir)

display(sf.numeric_epsilon)