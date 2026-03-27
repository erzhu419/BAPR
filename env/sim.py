import json
import time
import numpy as np
from env.timetable import Timetable
from env.bus import Bus
from env.route import Route
from env.station import Station
from env.visualize import visualize
import pandas as pd
from gym.spaces.box import Box
from gym.spaces import MultiDiscrete
import copy
import os, sys
import pygame
from collections import defaultdict
import json
import random

# ===== BA-PR: 导入模式配置 =====
from mode_profiles import MODE_PROFILES


class env_bus(object):
    
    def __init__(self, path, debug=False, render=False, route_sigma=1.5,
                 enable_mode_switch=False, mode_profiles=None,
                 mode_switch_interval=(1800, 7200)):
        """
        Args:
            path:                  环境数据目录
            debug:                 是否输出 debug 信息
            render:                是否渲染
            route_sigma:           路段速度的 lognormal sigma
            enable_mode_switch:    是否启用 BA-PR 模式切换
            mode_profiles:         模式配置字典, 默认使用 MODE_PROFILES
            mode_switch_interval:  模式切换间隔 (min_seconds, max_seconds)
        """
        if render:
            pygame.init()

        self.path = path
        self.route_sigma = float(route_sigma)
        sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            args = json.load(f)
        self.args = args
        self.effective_trip_num = 264
        
        self.time_step = args["time_step"]
        self.passenger_update_freq = args["passenger_state_update_freq"]
        # read data, multi-index used here
        self.od = pd.read_excel(os.path.join(path, "data/passenger_OD.xlsx"), index_col=[1, 0])
        self.station_set = pd.read_excel(os.path.join(path, "data/stop_news.xlsx"))
        self.routes_set = pd.read_excel(os.path.join(path, "data/route_news.xlsx"))
        self.timetable_set = pd.read_excel(os.path.join(path, "data/time_table.xlsx"))
        # Truncate the original timetable by first 50 trips to reduce the calculation pressure
        self.timetable_set = self.timetable_set.sort_values(by=['launch_time', 'direction'])[:self.effective_trip_num].reset_index(drop=True)
        # add index for timetable
        self.timetable_set['launch_turn'] = range(self.timetable_set.shape[0])
        self.max_agent_num = 25

        self.visualizer = visualize(self)

        # Set effective station and time period
        self.effective_station_name = sorted(set([self.od.index[i][0] for i in range(self.od.shape[0])]))
        self.effective_period = sorted(list(set([self.od.index[i][1] for i in range(self.od.shape[0])])))

        self.action_space = Box(0, 60, shape=(1,))

        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self.state_dim = 7 + len(self.routes)//2

        # ===== BA-PR: 模式切换参数 =====
        self.enable_mode_switch = enable_mode_switch
        self.mode_profiles = mode_profiles if mode_profiles is not None else MODE_PROFILES
        self.mode_switch_interval = mode_switch_interval
        self.mode_names = list(self.mode_profiles.keys())

        # BA-PR state (initialized in reset)
        self.current_mode_name = "normal"
        self.next_switch_time = 0
        self.mode_switch_count = 0
        self.mode_history = []

    # return the bus which is in terminal for now (which is not on route)
    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        return [
            Route(
                self.routes_set['route_id'][i],
                self.routes_set['start_stop'][i],
                self.routes_set['end_stop'][i],
                self.routes_set['distance'][i],
                self.routes_set['V_max'][i],
                self.routes_set.iloc[i, 5:],
                sigma=self.route_sigma
            )
            for i in range(self.routes_set.shape[0])
        ]

    def set_stations(self):
        station_concat = pd.concat([self.station_set, self.station_set[::-1][1:]]).reset_index()
        total_station = []
        for idx, station in station_concat.iterrows():
            # station type is 0 if Terminal else 1
            station_type = 1 if station['stop_name'] not in ['Terminal_up', 'Terminal_down'] else 0

            direction = False if idx >= station_concat.shape[0] / 2 else True
            od = None
            if station['stop_name'] in self.effective_station_name:
                od = self.od.loc[station['stop_name'], station['stop_name']:] if direction else self.od.loc[station['stop_name'], :station['stop_name']]
                # To reduce the OD value in False direction stations in ['X13','X14','X15'] because too many passengers stuck cause the overwhelming
                if station['stop_name'] in ['X13','X14','X15'] and not direction:
                    od *= 0.4

                od.index = od.index.map(str)
                od = od.to_dict(orient='index')

            total_station.append(Station(station_type, station['stop_id'], station['stop_name'], direction, od))

        return total_station

    # return default state and reward
    def reset(self):

        self.current_time = 0

        # initialize station, routes and timetables
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        # initial list of bus on route
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []

        # self.state is combine with route_state, which contains the route.speed_limit of each route, station_state, which
        # contains the station.waiting_passengers of each station and bus_state, which is bus.obs for each bus.
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.done = False

        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

        # ===== BA-PR: 重置模式状态 =====
        self.current_mode_name = "normal"
        self.mode_switch_count = 0
        self.mode_history = [("normal", 0)]
        if self.enable_mode_switch:
            self.next_switch_time = random.randint(*self.mode_switch_interval)
        else:
            self.next_switch_time = float('inf')
        self._apply_mode("normal")

    def initialize_state(self, render=False):
        def count_non_empty_sublist(lst):
            return sum(1 for sublist in lst if sublist)

        while count_non_empty_sublist(list(self.state.values())) == 0:
            self.state, self.reward, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    # ===== BA-PR: 模式切换引擎 =====
    def _apply_mode(self, mode_name: str):
        """
        将模式参数应用到环境的 routes 和 stations
        对应 BAPR_engineering_doc.md §2
        """
        profile = self.mode_profiles[mode_name]
        self.current_mode_name = mode_name

        # 应用路段参数
        affected_routes = profile.get("affected_routes", None)
        for i, route in enumerate(self.routes):
            if affected_routes is None or i in affected_routes:
                route.speed_mean_scale = profile["speed_mean_scale"]
                route.sigma = profile["sigma"]
                route.speed_cap = profile["speed_cap"]
            else:
                # 不受影响的路段恢复正常
                route.speed_mean_scale = 1.0
                route.sigma = self.route_sigma
                route.speed_cap = 15

        # 应用站点参数
        station_overrides = profile.get("station_od_overrides", {})
        for station in self.stations:
            if station.station_name in station_overrides:
                station.od_multiplier = station_overrides[station.station_name]
            else:
                station.od_multiplier = profile.get("od_global_mult", 1.0)

        # 立即刷新一次路段速度（避免等到下一个 route_state_update_freq）
        for route in self.routes:
            route.route_update(self.current_time, self.effective_period)

    def _maybe_switch_mode(self):
        """
        检查是否到了切换时间，如果是则随机选择一个新模式
        """
        if not self.enable_mode_switch:
            return

        if self.current_time >= self.next_switch_time:
            # 从所有模式中随机选一个（排除当前模式）
            candidates = [m for m in self.mode_names if m != self.current_mode_name]
            new_mode = random.choice(candidates) if candidates else self.current_mode_name
            self._apply_mode(new_mode)

            self.mode_switch_count += 1
            self.mode_history.append((new_mode, self.current_time))
            self.next_switch_time = self.current_time + random.randint(*self.mode_switch_interval)

    def get_current_mode_info(self) -> dict:
        """返回当前模式信息，供日志和 belief 计算使用"""
        return {
            "mode_name": self.current_mode_name,
            "mode_switch_count": self.mode_switch_count,
            "next_switch_time": self.next_switch_time,
            "mode_history": self.mode_history,
        }

    def launch_bus(self, trip):
        # Trip set(self.timetable) contain both direction trips. So we have to make sure the direction and launch time
        # is satisfied before the trip launched.
        # If there is no more appropriate bus in terminal, create a new bus, then add it to all_bus list.
        if len(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal))) == 0:
            # cause bus.next_station, current_route and effective station & routes is defined by @property, so no initialize here
            bus = Bus(self.bus_id, trip.launch_turn, trip.launch_time, trip.direction, self.routes, self.stations)
            self.bus_all.append(bus)
            self.bus_id += 1
        else:
            # if there is bus in terminal and also the direction is satisfied, then we reuse the bus to relaunch one of
            # them, which has the earliest arrived time to terminal.
            bus = sorted(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal)), key=lambda bus: bus.back_to_terminal_time)[0]
            bus.reset_bus(trip.launch_turn, trip.launch_time)
            # in drive() function, we set bus.on_route = False when it finished a trip. Here we set it to True because
            # the iteration in drive(), we just update the state of those bus which on routes
            bus.on_route = True

    def step(self, action, debug=False, render=False):
        # ===== BA-PR: 模式切换检查 =====
        self._maybe_switch_mode()

        # Enumerate trips in timetables, if current_time<=launch_time of the trip, then launch it.
        self.reward = defaultdict(float)
        for i, trip in enumerate(self.timetables):
            if trip.launch_time <= self.current_time and not trip.launched:
                trip.launched = True
                self.launch_bus(trip)
        # route
        route_state = []
        # update route speed limit by freq
        if self.current_time % self.args['route_state_update_freq'] == 0:
            for route in self.routes:
                route.route_update(self.current_time, self.effective_period)
                route_state.append(route.speed_limit)
            self.route_state = route_state
        # update waiting passengers of every station every second
        if self.current_time % self.passenger_update_freq == 0:
            for station in self.stations:
                station.station_update(self.current_time, self.stations, self.passenger_update_freq)
        # update bus state
        for bus in self.bus_all:
            bus.reward = None
            bus.obs = []
            if bus.in_station:
                bus.trajectory.append([bus.last_station.station_name, self.current_time, bus.absolute_distance, bus.direction, bus.trip_id])
                bus.trajectory_dict[bus.last_station.station_name].append([bus.last_station.station_name, self.current_time + bus.holding_time, bus.absolute_distance, bus.direction, bus.trip_id])
            if bus.on_route:
                bus.drive(self.current_time, action[bus.bus_id], self.bus_all, debug=debug)

        self.state_bus_list = state_bus_list = list(filter(lambda x: len(x.obs) != 0, self.bus_all))
        self.reward_list = reward_list = list(filter(lambda x: x.reward is not None, self.bus_all))

        if len(state_bus_list) != 0:
            for i in range(len(state_bus_list)):
                if state_bus_list[i].bus_id not in self.state:
                    self.state[state_bus_list[i].bus_id] = []
                self.state[state_bus_list[i].bus_id].append(state_bus_list[i].obs)
        if len(reward_list) != 0:
            for i in range(len(reward_list)):
                self.reward[reward_list[i].bus_id] = reward_list[i].reward

        self.current_time += self.time_step
        unhealthy_all = [bus.is_unhealthy for bus in self.bus_all]
        if sum([trip.launched for trip in self.timetables]) == len(self.timetables) and sum([bus.on_route for bus in self.bus_all]) == 0:
            self.done = True
            for bus in self.bus_all:
                bus.trajectory.clear()
                bus.trajectory_dict.clear()
                del bus.trajectory
                del bus.trajectory_dict
            for station in self.stations:
                station.waiting_passengers = np.array([])
                station.total_passenger.clear()
        else:
            self.done = False

        if self.done and debug:
            self.summary_data = self.summary_data.sort_values(['bus_id', 'time'])

            output_dir = os.path.join(self.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            self.visualizer.plot()

            self.summary_data.to_csv(os.path.join(output_dir, 'summary_data.csv'))
            self.summary_reward = self.summary_reward.sort_values(['bus_id', 'time'])
            self.summary_reward.to_csv(os.path.join(self.path, 'pic', 'summary_reward.csv'))

        if render and self.current_time % 1 == 0:
            self.visualizer.render()
            time.sleep(0.05)  # Add a delay to slow down the rendering

        return self.state, self.reward, self.done


if __name__ == '__main__':
    debug = False
    render = False

    env = env_bus(os.path.join(os.path.dirname(__file__)),
                  debug=debug,
                  enable_mode_switch=True,
                  mode_switch_interval=(500, 2000))  # 短间隔测试
    start_time = time.time()
    actions = {key: 15. for key in list(range(env.max_agent_num))}
    env.reset()
    print(f"Initial mode: {env.current_mode_name}, next switch at t={env.next_switch_time}")
    while not env.done:
        state, reward, done = env.step(action=actions, debug=debug, render=render)
    
    print(f"\nDone in {time.time() - start_time:.1f}s")
    print(f"Mode switches: {env.mode_switch_count}")
    print(f"Mode history:")
    for mode, t in env.mode_history:
        print(f"  t={t:6d}: {mode}")
