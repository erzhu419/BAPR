from env.passenger import Passenger
import numpy as np


class Station(object):
    def __init__(self, station_type, station_id, station_name, direction, od):
        # if the station is terminal or not terminal,
        self.station_type = station_type
        # the id of stations
        self.station_id = station_id
        self.station_name = station_name
        # waiting passengers in this station
        self.waiting_passengers = np.array([])
        self.total_passenger = []
        # the direction is True if upstream, else False
        self.direction = direction
        # od is the passengers demand of every hour
        self.od = od
        # ===== BA-PR: 新增 =====
        self.od_multiplier = 1.0  # 站点级 OD 倍率

    def station_update(self, current_time, stations, passenger_update_interval=1):
        """
        每秒更新一次，减少不必要的泊松分布计算
        """
        if self.od is not None:  # 确保存在OD矩阵
            effective_period_str = f"{6 + min(current_time // 3600, 13):02}:00:00"  # 每小时的有效时间段
            period_od = self.od[effective_period_str]  # 获取该时间段的OD需求

            # 计算每秒的平均需求
            for destination_name, demand in period_od.items():
                if demand > 0:
                    # ===== BA-PR: 乘以 OD 倍率 =====
                    demand_per_second = (demand * self.od_multiplier) / 3600.0  # 每秒的需求量

                    # 用每秒的需求量进行泊松分布采样
                    destination_demand_num = np.random.poisson(demand_per_second * passenger_update_interval)

                    if destination_demand_num > 0:
                        destination = next(
                            x for x in stations
                            if x.station_name == destination_name and x.direction == self.direction
                        )

                        # 创建新乘客并更新等候队列
                        new_passengers = [
                            Passenger(current_time, self, destination)
                            for _ in range(destination_demand_num)
                        ]
                        self.waiting_passengers = np.append(self.waiting_passengers, new_passengers)
                        self.total_passenger.extend(new_passengers)
