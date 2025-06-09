import os
import yaml
import pymap3d
import numpy as np


def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """
    计算 AO（我机对敌夹角）、TA（敌机对我夹角）和相对距离 R。

    参数：
        ego_feature: 我方特征，格式为 (north, east, up, vn, ve, vd)
        enm_feature: 敌方特征，格式相同
        return_side: 是否返回敌人在哪一侧（左 or 右）

    返回：
        ego_AO: Aspect Offset（我方对敌夹角）
        ego_TA: Target Aspect（敌方对我夹角）
        R: 两机距离（欧几里得距离）
        side_flag: （可选）敌机相对位置（1为右侧，-1为左侧）
    """

    # 分解我方位置和速度
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])  # 我机速度模长

    # 分解敌方位置和速度
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])  # 敌机速度模长

    # 计算我方与敌方的相对位置向量
    delta_x = enm_x - ego_x
    delta_y = enm_y - ego_y
    delta_z = enm_z - ego_z

    # 相对距离 R（欧几里得范数）
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    # -----------------------------------
    # 计算我方的 AO（Aspect Offset）
    # 即：我方速度矢量与我方指向敌人的向量之间的夹角
    # 公式：AO = arccos((Δ · V_ego) / (|Δ||V_ego|))
    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

    # 计算我方的 TA（Target Aspect）
    # 即：敌方速度矢量与敌方指向我方的向量之间的夹角
    # 公式：TA = arccos((Δ · V_enm) / (|Δ||V_enm|))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag




def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def get_thetaT_thetaA_R(ego_feature, enm_feature):
    """
        计算 追踪角thetaT、thetaA背向角和相对距离 R。

        参数：
            ego_feature: 我方特征，格式为 (north, east, up, vn, ve, vd, r, p, y)
            enm_feature: 敌方特征，格式相同
            return_side: 是否返回敌人在哪一侧（左 or 右）

        返回：
        thetaT: 追踪角（单位：弧度，范围 [0, π]）
        thetaA: 背向角（单位：弧度，范围 [0, π]）
        R: 两机之间的欧几里得距离
    """
    # 分解我方位置和速度
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz,ego_roll, ego_pitch, ego_yaw = ego_feature
    # ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])  # 我机速度模长

    # 分解敌方位置和速度
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz,enm_roll, enm_pitch, enm_yaw = enm_feature
    # enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])  # 敌机速度模长

    # 相对位置向量 Δ
    delta_x = enm_x - ego_x
    delta_y = enm_y - ego_y
    delta_z = enm_z - ego_z
    delta_vec = np.array([delta_x, delta_y, delta_z])
    R = np.linalg.norm(delta_vec)

    delta_unit = delta_vec / R

    # 计算我机朝向向量（由 pitch 和 yaw 确定）
    ego_dir = np.array([
        np.cos(ego_pitch) * np.cos(ego_yaw),
        np.cos(ego_pitch) * np.sin(ego_yaw),
        np.sin(ego_pitch)
    ])

    # 计算敌机尾部朝向向量（即负的敌机朝向方向）
    enm_dir = -np.array([
        np.cos(enm_pitch) * np.cos(enm_yaw),
        np.cos(enm_pitch) * np.sin(enm_yaw),
        np.sin(enm_pitch)
    ])

    # 追踪角 thetaT：ego 朝向 与 Δ 的夹角
    dot_thetaT = np.dot(ego_dir, delta_unit)
    thetaT = np.arccos(np.clip(dot_thetaT, -1, 1))

    # 背向角 thetaA：enm 尾部方向 与 -Δ 的夹角（即朝向 ego）
    dot_thetaA = np.dot(enm_dir, -delta_unit)
    thetaA = np.arccos(np.clip(dot_thetaA, -1, 1))

    return thetaT, thetaA, R

def S(x, alpha, x0):
    return 1 / (1 + np.exp(-alpha * (x - x0)))

def get_forward_vector_from_rpy(rpy):
    roll, pitch, yaw = rpy
    # 只考虑 yaw 和 pitch 对朝向的影响
    cx = np.cos(pitch)
    sx = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # 飞行器的机头方向（前向向量）
    forward = np.array([
        cx * cy,  # x (前向)
        cx * sy,  # y
        sx        # z (俯仰)
    ])
    return forward / (np.linalg.norm(forward) + 1e-6)