import math
import random
import numpy as np

class Route(object):
    def __init__(self, route_id, start_stop, end_stop, route_length, max_speed, route_speed_history, sigma=1.5):
        self.route = []
        self.maximum_velocity = 0
        self.variant_velocity = 0

        self.sigma = float(sigma)
        self.route_id = route_id
        self.route_max_speed = max_speed
        self.speed_history = route_speed_history
        self.speed_limit = 15

        self.start_stop = start_stop
        self.end_stop = end_stop
        self.distance = route_length

        # ===== BA-PR: 新增属性 =====
        self.base_max_speed = max_speed          # 原始最大速度（保存）
        self.speed_mean_scale = 1.0              # 均值缩放比例（线性空间）
        self.speed_cap = 15                      # 动态速度上限

    def route_update(self, current_time, effective_period):
        current_hour = effective_period[min(current_time//3600, len(effective_period) -1)]
        # ===== BA-PR: 均值缩放 =====
        # lognormvariate(mu, sigma) 的 mu 是对数域参数
        # 要在线性空间实现 speed_mean_scale 倍缩放，需在对数域做加法偏移
        base_mu = self.speed_history.loc[current_hour]
        scaled_mu = base_mu + math.log(self.speed_mean_scale)  # log域偏移 = 线性域缩放
        v = np.clip(math.log(random.lognormvariate(scaled_mu, self.sigma)), 2, self.speed_cap)
        self.speed_limit = min(self.speed_cap, max(int(v), 0))
