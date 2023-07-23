import datetime
import math
import random

import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

from util.log_render import render_to_file
from util.plot_chart import TradingChart
from util.read_config import EnvConfig
from Data_train.dataproc import dataproc
import MetaTrader5 as mt5
import json
import pickle


class tgym(gym.Env):
    

    metadata = {"render.modes": ["graph", "human", "file", "none"]}

    def __init__(self, leverage,balance, equity, AI_filter=False, num_rows=None, df=[],training=True) -> None:
        super(tgym, self).__init__()
        print("TRADING ENVIRONMENT INITIALISED\n\n")
        if len(df) == 0:
            path = r'C:\Users\Huzaifah-Admin\Desktop\cam\FinRL-Meta\meta\env_fx_trading\Data_train\\'
        else:
            path = None
            
        self.training=training

        config_file_path = "config/gdbusd-test-1.json"
        parameters_to_be_scaled = []

        with open('Data_train\symbol_encoding.pkl', 'rb') as file:
            self.symbol_encoding = pickle.load(file)

        with open(config_file_path, 'r') as file:
            config_data = json.load(file)

        symbol_keys = list(config_data['symbol'].keys())
        parameters_for_pairs = [
            "max_spread", "transaction_fee", "over_night_penalty", "stop_loss_max"]
        parameters_to_be_scaled = [config_data['symbol'][i][j]
                                   for i in symbol_keys for j in parameters_for_pairs]
        
        if training:
            row_control=500
        else:
            row_control=None

        data_processor = dataproc(df=df, path=path, row_control=row_control)
        transformed_param = data_processor.scale_config(
            parameters_to_be_scaled)

        for i in symbol_keys:
            for j in parameters_for_pairs:
                config_data['symbol'][i][j] = transformed_param[parameters_for_pairs.index(
                    j)][0]

        df = data_processor.get_df(
            ["open", "spread", '1', '2', '3', '4', '5', '6'], (-2, 2), False, False)
        df_columns = df.columns.tolist()
        config_data["env"]['observation_list'] = df_columns

        with open("config/config_new.json", "w") as file:
            json.dump(config_data, file, indent=4)

        env_config_file = 'config/config_new.json'

        assert df.ndim == 2

        self.AI_filter = AI_filter
        self.leverage = leverage
        self.balance_initial = balance
        self.equity = equity

        
        self.cf = EnvConfig(env_config_file)
        self.observation_list = self.cf.env_parameters("observation_list")

        self.over_night_cash_penalty = self.cf.env_parameters(
            "over_night_cash_penalty")
        self.asset_col = self.cf.env_parameters("asset_col")

        self.time_col = self.cf.env_parameters("time_col")
        
        if training:
            self.random_start = self.cf.env_parameters("random_start")
        else:
            self.random_start=0
        
        self.log_filename = (
            self.cf.env_parameters("log_filename")
            + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            + ".csv"
        )

        self.df = df
        self.df["_time"] = df[self.time_col]
        self.time_col = "_time"
        self.df["_day"] = df["weekday"]
        self.assets = df[self.asset_col].unique()
        with open('assets.pkl', 'wb') as f:
            pickle.dump(self.assets,f) 
        
        self.points_list=self.get_symbol_info('point')


        self.dt_datetime = df[self.time_col].sort_values().unique()
        print("DATETIME LIST: ", self.dt_datetime)
        self.df = self.df.set_index(self.time_col)
        self.visualization = True

        # --- reset value ---
        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.equity
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.current_step = 0
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        # --- end reset ---

        self.cached_data = [
            self.get_observation_vector(_dt) for _dt in self.dt_datetime
        ]
        self.cached_time_serial = (
            (self.df[["time", "_day"]].sort_values("time")).drop_duplicates()
        ).values.tolist()
        self.stop_loss_max = 0
        self.profit_taken_max = 0
        self.position_sizing = 0

        if AI_filter:

            self.low = np.array([0]*(len(self.assets)+3))
            self.high = np.array([1]*(len(self.assets)+3))
            self.high[-3:] = 1000000
            self.action_space = spaces.Box(
                low=self.low, high=self.high, shape=(len(self.assets)+3,))

        else:

            self.low = np.array([0, 0, 0])
            self.high = np.array([1000000, 1000000, 1000000])
            self.action_space = spaces.Box(
                low=self.low, high=self.high, shape=(3,))

        self.reward_range = (-np.inf, np.inf)

        # first two 3 = balance,current_holding, max_draw_down_pct
        _space = 3 + len(self.assets) + len(self.assets) * \
            len(self.observation_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_space,))
        print(
            f"initial done:\n"
            f"observation_list:{self.observation_list}\n "
            f"assets:{self.assets}\n "
            f"time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}"
        )
        self._seed()

    def pad_sublists(self, array, length=20):
        max_length = length
        padded_array = [sublist + [0] *
                        (max_length - len(sublist)) for sublist in array]
        return padded_array

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _history_df(self, i):
        pass

    def _take_action(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max )
        # _actions = np.floor(actions).astype(int)
        # _profit_takens = np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        _action = 2
        _profit_taken = 0
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        # need use multiply assets
        top_gainer_actions = [0, 2]
        top_loser_actions = [1, 2]

        self.position_sizing, self.profit_taken_max, self.stop_loss_max = actions[-3:]
        actions = actions[:-3]

        for i, x in enumerate(actions):
            #self._o = self.get_observation(self.current_step, i, "open")
            self._h = self.get_observation(self.current_step, i, "high")
            self._l = self.get_observation(self.current_step, i, "low")
            self._c = self.get_observation(self.current_step, i, "close")
            self._t = self.get_observation(self.current_step, i, "time")
            self._day = self.get_observation(self.current_step, i, "_day")
            self._top_gainer = self.get_observation(
                self.current_step, i, "Top_Gainer")
            self._top_loser = self.get_observation(
                self.current_step, i, "Top_Loser")

            _action = math.floor(x)
            rewards[i] = self._calculate_reward(i, done)
            if True:
                self._limit_order_process(i, _action, done)
            if (
                _action in (0, 1)
                and not done
                and self.current_holding[i]
                < 200
            ):
                # generating PT based on action fraction
                _profit_taken = math.ceil(
                    (x - _action) *
                    self.profit_taken_max
                ) + self.stop_loss_max
                self.ticket_id += 1
                if False:
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": self.position_sizing,
                        "ActionPrice": self._l if _action == 0 else self._h,
                        "SL": self.stop_loss_max,
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -0.02,
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": -1,
                        "CloseStep": -1,
                    }
                    self.transaction_limit_order.append(transaction)
                else:
                    if self._top_gainer:
                        _action = top_gainer_actions[_action]


                    elif self._top_loser:
                        _action = top_gainer_actions[_action]

                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": self.position_sizing,
                        "ActionPrice": self._c,
                        "SL": self.stop_loss_max,
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": self.points_list[i],
                        "Reward": -10,
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= 10
                    self.transaction_live.append(transaction)

        return sum(rewards)

    def _calculate_reward(self, i, done):
        _total_reward = 0
        _max_draw_down = 0
        for tr in self.transaction_live:
            if tr["Symbol"] == self.assets[i]:
                _point = tr['Point']
                # cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.symbol(self.assets[i],
                                                   "over_night_penalty")

                if tr["Type"] == 0:  # buy
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                    if done:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._l <= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._h >= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int(
                            (self._l - tr["ActionPrice"]) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == 1:  # sell
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                    if done:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._h >= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._l <= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward

    def _take_action_EA(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max )
        # _actions = np.floor(actions).astype(int)
        # _profit_takens = np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        _action = 0
        _profit_taken = 0
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        # need use multiply assets

        self.position_sizing, self.profit_taken_max, self.stop_loss_max = actions[:]


        for i in range(len(self.assets)):
            #self._o = self.get_observation(self.current_step, i, "open")
            self._h = self.get_observation(self.current_step, i, "high")
            self._l = self.get_observation(self.current_step, i, "low")
            self._c = self.get_observation(self.current_step, i, "close")
            self._t = self.get_observation(self.current_step, i, "time")
            self._day = self.get_observation(self.current_step, i, "_day")
            self._top_gainer = self.get_observation(
                self.current_step, i, "Top_Gainer")
            self._top_loser = self.get_observation(
                self.current_step, i, "Top_Loser")

            rewards[i] = self._calculate_reward(i, done)
            if True:
                self._limit_order_process(i, _action, done)
            if (
                _action in (0, 1)
                and not done
                and self.current_holding[i]
                < 200
            ):

                if False:
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": self.position_sizing,
                        "ActionPrice": self._l if _action == 0 else self._h,
                        "SL": self.stop_loss_max,
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -0.02,
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": -1,
                        "CloseStep": -1,
                    }
                    self.transaction_limit_order.append(transaction)
                else:
                    # generating PT based on action fraction
                    _profit_taken = math.ceil(
                        1 *
                        self.profit_taken_max
                    ) + self.stop_loss_max
                    self.ticket_id += 1

                    if self._top_gainer:
                        _action = 0


                    elif self._top_loser:
                        _action = 1
           

                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": self.position_sizing,
                        "ActionPrice": self._c,
                        "SL": self.stop_loss_max,
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -10,
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= 10
                    self.transaction_live.append(transaction)

        return sum(rewards)

    def _limit_order_process(self, i, _action, done):
        for tr in self.transaction_limit_order:
            if tr["Symbol"] == self.assets[i]:
                if tr["Type"] != _action or done:
                    self.transaction_limit_order.remove(tr)
                    tr["Status"] = 3
                    tr["CloseStep"] = self.current_step
                    self.transaction_history.append(tr)
                elif (tr["ActionPrice"] >= self._l and _action == 0) or (
                    tr["ActionPrice"] <= self._h and _action == 1
                ):
                    tr["ActionStep"] = self.current_step
                    self.current_holding[i] += 1
                    self.balance -= 10
                    self.transaction_limit_order.remove(tr)
                    self.transaction_live.append(tr)
                    self.tranaction_open_this_step.append(tr)
                elif (
                    tr["LimitStep"]
                    + 5.0
                    > self.current_step
                ):
                    tr["CloseStep"] = self.current_step
                    tr["Status"] = 4
                    self.transaction_limit_order.remove(tr)
                    self.transaction_history.append(tr)

    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = self._t
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

    def step(self, actions):
        # Execute one time step within the environment
        self.current_step += 1
        done = self.balance <= 0 or self.current_step == len(
            self.dt_datetime) - 1
        if done:
            self.done_information += f"Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n"
            self.visualization = True

        if self.AI_filter:
            reward = self._take_action(actions, done)

        else:
            reward = self._take_action_EA(actions, done)
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
        if self.balance != 0:
            self.max_draw_down_pct = abs(
                sum(self.max_draw_downs) / self.balance * 100)

            # no action anymore
        obs = (
            [self.balance, self.max_draw_down_pct, self.total_equity]
            + self.current_holding
            + self.get_observation(self.current_step)
        )

        obs = [timestamp.timestamp() if isinstance(timestamp, pd.Timestamp)
               else timestamp for timestamp in obs]
        obs = [self.encode_categorical(value) if isinstance(
            value, str) else float(value) for value in obs]
        return (
            np.array(obs).astype(np.float32),
            reward,
            done,
            {"Close": self.tranaction_close_this_step},
        )

    def get_observation(self, _step, _iter=0, col=None):
        if col is None:
            return self.cached_data[_step]
        if col == "_day":
            return self.cached_time_serial[_step][1]

        elif col == "time":
            return self.cached_time_serial[_step][0]
        col_pos = -1
        for i, _symbol in enumerate(self.observation_list):
            if _symbol == col:
                col_pos = i
                break

        return self.cached_data[_step][_iter * len(self.observation_list) + col_pos]

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list
        
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"'
                )
            if subset.empty:
                # Handle the case when the subset DataFrame is empty
                v += [0] * len(cols)  # Append zeros to the vector
            else:
                v += subset.loc[_dt, cols].values.tolist()

        if len(v) != len(self.assets) * len(cols):
            # Handle the case when the resulting vector length is incorrect
            print("Length of the observation vector is incorrect")
            return None  # Or raise an exception, depending on your use case

        return v


    def reset(self):
        # Reset the state of the environment to an initial state
        self.seed()

        if self.random_start:
            self.current_step = random.choice(
                range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        self.visualization = False
        
        current_step_obs=self.get_observation(self.current_step)
        
        print("CURRENT OBS STEP: ",len(current_step_obs))
        print("ASSETS LENGTH: ",len(self.assets))

        observation = [
            self.balance, self.max_draw_down_pct, self.total_equity
        ] + [0] * len(self.assets) + current_step_obs

        # Convert Timestamp to float
        observation = [timestamp.timestamp() if isinstance(
            timestamp, pd.Timestamp) else timestamp for timestamp in observation]

        
        observation = [
            self.encode_categorical(value) if isinstance(value, str) else float(value) if pd.notnull(value) else 0
            for value in observation
            ]   
        
        print("Observation shape:", len(observation))

        return np.array(observation).astype(np.float32)

    def encode_categorical(self, value):
        # Use your encoding dictionary to encode categorical variables
        # Replace `self.symbol_encoding` with your actual encoding dictionary
        encoded_value = self.symbol_encoding.get(value, 0)
        return encoded_value

    def render(self, mode="human", title=None, **kwargs):
        # Render the environment to the screen
        if mode in ("human", "file"):
            printout = mode == "human"
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information,
            }
            render_to_file(**pm)
            if self.log_header:
                self.log_header = False
        elif mode == "graph" and self.visualization:
            print("plotting...")
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass
    
    def get_symbol_info(self, prop):
        symbol_property = []

        for symbol in self.assets:
            symbol_info = mt5.symbol_info(symbol)
            value = getattr(symbol_info, prop)
            symbol_property.append(value)

        return symbol_property
        
        
        
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# technical features
# replay_buffer
# observation_list dynamic
# config file normalisation
# streaming
# deployment file.
