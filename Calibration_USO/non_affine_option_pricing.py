import numpy as np
import scipy.integrate as integrate
from numba import jit, int32, float64


def call_option_price(rn_params, K, Y_0, V_0, t, s, v_st, p_params):
    f = lambda u: (np.exp(-1j * u * np.log(K))*char_fun((u - 0.5j)/100, Y_0, V_0, t, s, v_st, params)/(u**2 + 0.25)).real
    return char_fun(-1j/100, Y_0, V_0, t, s, v_st, params).real - np.sqrt(K)/np.pi*integrate.quad(lambda u: f(u), 10e-5, 100)[0]


#def put_option_price(K, Y_0, V_0, t, s, v_st, params):
#    f = lambda u: (K**(u*1j)*char_fun(-u + 0.5j, Y_0, V_0, t, s, v_st, params)/(u - 0.5j)/(u - 1.5j)).real
#    return  - 0.5*np.sqrt(K**3)/np.pi*integrate.quad(lambda u: f(u), -np.inf, np.inf)[0]




