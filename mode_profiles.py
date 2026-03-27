# Mode profiles for BA-PR SAC environment
# Each mode defines a distinct environmental regime with different parameters

MODE_PROFILES = {
    # ─── 正常模式 ───
    "normal": {
        "speed_mean_scale": 1.0,      # 速度均值不变
        "sigma": 1.5,                  # 标准方差
        "speed_cap": 15,               # 正常限速
        "od_global_mult": 1.0,         # 标准客流
        "station_od_overrides": {},     # 无站点特殊干预
        "affected_routes": None,       # 所有路段
    },

    # ─── 严重拥堵：某段道路事故/施工 ───
    "congestion_severe": {
        "speed_mean_scale": 0.3,       # 均值降到 30%
        "sigma": 3.0,                  # 波动也变大
        "speed_cap": 5,                # 限速降到 5 m/s
        "od_global_mult": 1.0,
        "station_od_overrides": {},
        "affected_routes": [3, 4, 5, 6],  # 只影响路段 3-6（中段路网）
    },

    # ─── 客流激增：学校放学 / 大型活动 ───
    "demand_surge": {
        "speed_mean_scale": 1.0,
        "sigma": 1.5,
        "speed_cap": 15,
        "od_global_mult": 1.5,         # 全局 1.5 倍
        "station_od_overrides": {      # 特定站点 OD 暴涨
            "X05": 5.0,                # 站点 X05 客流 5 倍
            "X06": 4.0,
            "X07": 3.0,
        },
        "affected_routes": None,
    },

    # ─── 全线瘫痪：极端天气 ───
    "extreme_weather": {
        "speed_mean_scale": 0.4,       # 全线均值降 60%
        "sigma": 4.0,                  # 极大方差
        "speed_cap": 8,                # 全线限速
        "od_global_mult": 0.3,         # 乘客也减少
        "station_od_overrides": {},
        "affected_routes": None,
    },

    # ─── 局部路段封闭 + 周边客流转移 ───
    "partial_closure": {
        "speed_mean_scale": 0.15,      # 封闭路段速度降到 15%
        "sigma": 1.0,
        "speed_cap": 3,                # 封闭路段限速 3
        "od_global_mult": 1.0,
        "station_od_overrides": {
            "X08": 4.0,
            "X09": 4.0,
            "X10": 3.0,
        },
        "affected_routes": [7, 8, 9],
    },
}
