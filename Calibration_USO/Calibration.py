import os
import csv
import pandas as pd
import Option_pricer_gpt
import numpy as np
import time

from numba import njit, types, prange
from numba.typed import Dict
from scipy.optimize import minimize



def timer_decorator(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{function.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper


class Calibration:
    """This class is used to calibrate the model"""

    def __init__(self, option_file: str, model_type=('SV', 'LIN', 0.5), rn_params=[0, 0, 0, 0, 0]):
        self.model_type = model_type
        self.option_file = option_file
        self._load_model(rn_params)
        #self.p_params = self._load_P_params()
        #self.rn_params = self._load_rn_params(rn_params)
        print(self.rn_params)

    def _load_model(self, rn_params):
        self.p_params = self._load_P_params()
        P_vols = self._load_P_vols()
        option_data_temp = self._load_option_data()
        self._append_P_vols_to_options_data(P_vols, option_data_temp)
        self.rn_params = self._load_rn_params(rn_params)

    def _load_option_data(self):
        option_data = pd.read_csv(self.option_file, header=0, index_col=None)
        option_data['Date'] = pd.to_datetime(option_data['Date'])
        option_data = option_data.drop(columns=['RIC', 'American_price'])
        return option_data

    def _load_P_params(self):
        path_parameters_folder = 'C:\\Users\\pawon\\Dropbox\\Option_pricing_USO\\Data\\Processed\\Parameters'
        file = self._gen_save_str(path_parameters_folder)
        names_params = ['mu', 'sigv', 'rho', 'a0', 'a1', 'a2', 'a3', 'b', 'lambda', 'muy', 'sigy', 'muv']
        df = pd.read_csv(file, header=None, index_col=None, names=names_params)

        if self.model_type[0] == 'SVCJ':
            param_dict = df.to_dict(orient='records')[0]
        elif self.model_type[0] == 'SVJ':
            param_dict = df[names_params[:-1]].to_dict(orient='records')[0]
        else:
            param_dict = df[names_params[:-4]].to_dict(orient='records')[0]

        numba_dict = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

        for key, value in param_dict.items():
            numba_dict[key] = value

        return numba_dict

    def _load_rn_params(self, rn_params):
        param_dict = {
            'lambda0y': rn_params[0],
            'lambda1y': rn_params[1],
            'lambda0v': rn_params[2],
            'lambda2v': rn_params[3],
            'lambda_jumps': rn_params[3]
        }

        numba_dict = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

        for key, value in param_dict.items():
            numba_dict[key] = value

        return numba_dict

    def update_rn_params(self, update_rn_params_list):
        keys = ['lambda0y', 'lambda1y', 'lambda0v', 'lambda2v', 'lambda_jumps']
        for i, key in enumerate(keys):
            self.rn_params[key] = update_rn_params_list[i]

    def _load_P_vols(self):
        '''
        Loads model volatilities estimated from the model that was estimated from PMCMC.
        '''
        path_vol_folder = 'C:\\Users\\pawon\\Dropbox\\Option_pricing_USO\\Data\\Processed\\Volatilities\\1July2007_31June2018'
        file = self._gen_save_str(path_vol_folder)
        volatilities = pd.read_csv(file, header=0, index_col=None)
        volatilities['Date'] = pd.to_datetime(volatilities['Date'])

        return volatilities

    def _append_P_vols_to_options_data(self, P_vols, option_data):
        join_option_volatility_data = pd.merge(option_data, P_vols, on='Date', how='outer').dropna()
        #These conditions are filters on which options we are calibrating on
        mask = (join_option_volatility_data['Type'] == 'C') & \
               (join_option_volatility_data['TTM'] <= 60 / 365) & \
               (join_option_volatility_data['Strike'] >= join_option_volatility_data['Underlying'])
        df = join_option_volatility_data[mask]
        df = df.drop(columns=['Date', 'Expiry', 'b'])
        df = df.dropna()
        dtype = np.dtype([
            ('Strike', float),
            ('Type', 'S1'),
            ('Underlying', float),
            ('r', float),
            ('TTM', float),
            ('European_price', float),
            ('IV', float),
            ('Vol', float)
        ])
        self.options_data = df.to_records(index=False).astype(dtype)

    def _gen_save_str(self, path):
        if self.model_type[2] == 1:
            str_file = f'{self.model_type[0]}_{self.model_type[1]}_b.1_USO_OVX_1July2007_31June2018_100Particles80000reps.csv'
        else:
            str_file = f'{self.model_type[0]}_{self.model_type[1]}_b.{self.model_type[2]}_USO_OVX_1July2007_31June2018_100Particles80000reps.csv'
        
        return os.path.join(path, str_file)

    def calibrate(self):
        initial_rn_params = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = minimize(Option_pricer_gpt.objective_function_wrapper, initial_rn_params, args=(self.options_data, self.p_params), method='Nelder-Mead')
        self.update_rn_params(result.x)
        return result.x

    def option_test(self):
        #Test function to see Market vs model price
        for i in range(self.options_data.shape[0]):
          row = self.options_data[i]
          K = row['Strike']
          Y_0 = np.log(row['Underlying']) * 100
          V_0 = row['Vol']
          r = row['r']
          flag = 'c' if row['Type'] == 'C' else 'p'
          s = row['TTM'] * 365
          v_st = V_0
          observed_price = row['European_price']
          model_price = Option_pricer_gpt.call_option_price(self.rn_params, K, Y_0, V_0, 0, s, v_st, self.p_params)
          print(f'Market Price {observed_price} Model Price {model_price}')


