import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from statistics import median
import datetime

class ACEnv(gym.Env):
    """Airconditioner environment for OpenAI gym"""

    def __init__(self, df, penalty_temp, penalty_energy, penalty_ope, prefered_temp_min, prefered_temp_max, max_limit):
        super(ACEnv, self).__init__()

        self.df = df
        self.penalty_temp = penalty_temp
        self.penalty_energy = penalty_energy
        self.penalty_ope = penalty_ope
        
        self.prefered_temp_min = prefered_temp_min
        self.prefered_temp_max = prefered_temp_max
        self.prefered_ope = 3
        
        self.max_limit = max_limit
        
#         try:
#             self.ac_col = [c for c in self.df.columns if 'heat_pump' in c][0]
#         except:
#             print("No AC data.")

        # Actions: Turn off or 5 different consumption level
        self.action_space = spaces.Discrete(6,)

        # stands for price, import generation 
#         self.observation_shape = (600, 800, 1)
#         self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
#                                             high = np.ones(self.observation_shape),
#                                             dtype = np.float16)

    def get_action_meanings(self, action):
        action_dic = {0: "Turn off", 1: "level 1", 2: "level 2", 3: "level 3", 4: "level 4", 5: "level 5"}
        return action_dic[action]
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self.temp = random.randrange(0, 200)
        self.done = False
        self.reward = 0
        self.action = 0
        self.energy_cost = 0
        self.total_energy_cost = 0
        self.time = 0
        self.current_step = 0
        
        self.net_energy_cost = 0
        self.total_net_energy_cost = 0
        
        # pick step from noon
#         df_noon = self.df[self.df['utc_timestamp'].dt.hour == 12]
#         rand = random.randint(0, df_noon.shape[0]-1)
#         self.current_step = list(df_noon['step'])[rand]

        self.cur_date = self.random_date(n=1) 
        self.fixed_cost = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['fixed'].values[0]
        self.generation = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['generation'].values[0]
        self.price = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['price'].values[0]      
        
        self.state = [self.time, self.price, self.generation, self.fixed_cost, self.total_energy_cost, self.current_step]
        
#         obs = np.array(self.energy_cost)
#         reward = np.array(self.reward)
        
        return np.array(self.state, dtype=np.float32)
    
    def get_reward(self):
        net_energy_cost_reward = self.price * self.net_energy_cost

        if action != 0:
            if self.total_net_energy_cost > self.max_limit:
                if self.temp < self.prefered_temp_min:
                    return -(net_energy_cost_reward + self.penalty_temp * (self.prefered_temp_min - self.temp) + self.penalty_energy*(self.total_energy_cost - self.max_limit) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
                elif self.temp > self.prefered_temp_max:
                    return -(net_energy_cost_reward + self.penalty_temp * (self.temp - self.prefered_temp_max) + self.penalty_energy*(self.total_energy_cost - self.max_limit) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
                else:
                    return -(net_energy_cost_reward + self.penalty_energy * (self.total_energy_cost - self.max_limit) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
            else:
                if self.temp < self.prefered_temp_min:
                    return -(net_energy_cost_reward + self.penalty_temp * (self.prefered_temp_min - self.temp) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
                elif self.temp > self.prefered_temp_max:
                    return -(net_energy_cost_reward + self.penalty_temp * (self.temp - self.prefered_temp_max) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
                else:
                    return -(net_energy_cost_reward + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
        else:
            if self.total_net_energy_cost > self.max_limit:
                return -(net_energy_cost_reward + self.penalty_energy*(self.total_energy_cost - self.max_limit) + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
            else:
                return -(net_energy_cost_reward + self.penalty_ope * abs((self.current_step - self.prefered_ope)))
                

    def step(self, action):
        self.action = action
        
        # set the price to the current step
#         self.price = float(self.df[self.df.step == self.current_step]['Day-ahead Price [EUR/MWh]'])
        # get consumption unit for unit energy consumption
#         consumption_unit = median(pd.to_numeric(self.df[self.ac_col].dropna()))/2.5
        consumption_unit = np.random.uniform(0.1,0.5)
        
        self.start_temp = self.temp
        # if the ac is not turned off
        if action != 0:
            # update the energy cost
            self.energy_cost = consumption_unit*action
            
            # update temperature based on the level
            if self.temp < self.prefered_temp_min:
                self.temp += action*2
            elif self.temp > self.prefered_temp_max:
                self.temp -= action*2
            else:
                self.temp += [-action*2, action*2][random.randint(0, 1)]
        else:
            self.energy_cost == 0
        self.total_energy_cost += self.energy_cost
        self.net_energy_cost = self.fixed_cost + self.energy_cost - self.generation
        
        if self.net_energy_cost < 0:
            self.net_energy_cost = 0
            
        self.total_net_energy_cost += self.net_energy_cost
    
        # calculate the reward and update reward 
        self.reward = float(self.get_reward())
        
        self.time += 1
        self.current_step += 1
        
        if self.time == 24:
            # if time = 24, we need to update using values at time = 0 in the following day.
            self.price = self.df[(self.df.date == datetime.datetime.strftime(pd.to_datetime(self.cur_date) + datetime.timedelta(days=1), '%Y-%m-%d')) & (self.df.t == 0)]['price'].values[0]
            self.generation = self.df[(self.df.date == datetime.datetime.strftime(pd.to_datetime(self.cur_date) + datetime.timedelta(days=1), '%Y-%m-%d')) & (self.df.t == 0)]['generation'].values[0]
            self.fixed_cost = self.df[(self.df.date == datetime.datetime.strftime(pd.to_datetime(self.cur_date) + datetime.timedelta(days=1), '%Y-%m-%d')) & (self.df.t == 0)]['fixed'].values[0]
            self.done = True
        else:
            self.price = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['price'].values[0]
            self.generation = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['generation'].values[0]
            self.fixed_cost = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time)]['fixed'].values[0]
            
        self.state = [self.time, self.price, self.generation, self.fixed_cost, self.total_energy_cost, self.current_step]
        return self.state, self.reward, self.done, {}


    def render(self, action):
        print('\n')
        if self.time - 1 >= 12:
            print(f'Time: {str(self.time - 1) + " PM"}')
        else:
            print(f'Time: {str(self.time - 1) + " AM"}')
        print(f'Start Temp: {str(self.start_temp) + " °C"}')
        print(f'Over Temp: {str(self.temp) + " °C"}')
        p = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time - 1)]['price'].values[0]
        g = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time - 1)]['generation'].values[0]
        c = self.df[(self.df.date == self.cur_date) & (self.df.t == self.time - 1)]['fixed'].values[0]
        print(f'Price: {p:.3f}')
        print(f'Generation:{g:.3f}')
        print(f'Fixed_cost for all other appliances:{c:.3f}')
        print(f'Action: {self.get_action_meanings(action)}')
        print(f'Current energy cost of AC:{self.energy_cost:.3f}')
        print(f'Total Energy Cost of AC: {self.total_energy_cost:.3f}') 
        print(f'Current net energy cost:{self.net_energy_cost:.3f}')
        print(f'Total net energy cost:{self.total_net_energy_cost:.3f}')
                                                                  
        print(f'Reward: {self.reward:.3f}')

    def random_date(self, n, seed=None):
        start = pd.to_datetime(self.df.date).min()
        end = pd.to_datetime(self.df.date).max() + datetime.timedelta(days=-1) # remove the last day
        ndays = (end - start).days + 1
        return datetime.datetime.strftime((pd.to_timedelta(np.random.rand(n) * ndays, unit='D') + start)[0], '%Y-%m-%d')
        
        