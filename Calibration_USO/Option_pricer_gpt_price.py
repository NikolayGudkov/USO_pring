import numpy as np
import scipy.integrate as integrate
from numba import jit, int32, float64, njit, prange, types
from numba.typed import Dict


def objective_function_wrapper(rn_params_array, options_data, p_params):
    keys = ['lambda0y', 'lambda1y', 'lambda0v', 'lambda2v', 'lambda_jumps']
    rn_params = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for i, key in enumerate(keys):
        rn_params[key] = rn_params_array[i]

    return objective_function_numba(rn_params, options_data, p_params)


@njit(parallel=True)
def objective_function_numba(rn_params: Dict, options_data, p_params: Dict):
    squared_diff_sum = 0
    for i in prange(options_data.shape[0]):
        row = options_data[i]
        K = row['Strike']
        Y_0 = np.log(row['Underlying']) * 100
        V_0 = row['Vol']
        s = row['TTM'] * 365
        v_st = V_0

        observed_price = row['European_price']
        model_price = call_option_price(rn_params, K, Y_0, V_0, 0, s, v_st, p_params)

        squared_diff_sum += (observed_price - model_price) ** 2
    print(squared_diff_sum)
    return squared_diff_sum

@njit
def integrand(u, K, Y_0, V_0, t, s, v_st, p_params, rn_params):
    f = (np.exp(-1j * u * np.log(K)) * char_fun((u - 0.5j) / 100, Y_0, V_0, t, s, v_st, p_params, rn_params) / (u ** 2 + 0.25)).real
    return f


@njit
def simpsons_rule_integration(integrand, a, b, num_points, *args):
    if num_points % 2 == 1:
        num_points += 1
    h = (b - a) / num_points
    x = np.linspace(a, b, num_points + 1)
    y = integrand(x, *args)
    integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
    return integral


@njit
def call_option_price(rn_params, K, Y_0, V_0, t, s, v_st, p_params):
    integral = simpsons_rule_integration(integrand, 10e-5, 60, 10000, K, Y_0, V_0, t, s, v_st, p_params, rn_params)
    return char_fun(-1j / 100, Y_0, V_0, t, s, v_st, p_params, rn_params).real - np.sqrt(K) / np.pi * integral


@njit
def char_fun(phi, y, v, t, s, v_st, p_params, rn_params):
    tau = s - t
    return np.exp(1j*phi*y + v*A(tau, p_params, rn_params, phi, v_st) + B(tau, p_params, rn_params, phi, v_st))


@njit
def omega_v(v,p_params, rn_params):
    return (p_params['a0'] + rn_params['lambda0v']) + p_params['a1']/v + (p_params['a2'] + rn_params['lambda2v'])*v + p_params['a3']*v**2


@njit
def omega_vy(v, p_params, rn_params):
    return v**(p_params['b'] + 0.5)


@njit
def omega_vv(v, p_params, rn_params):
    return v**(2 * p_params['b'])


@njit
def omega_v_der(v, p_params, rn_params):
    return -p_params['a1']/v**2 + (p_params['a2'] + rn_params['lambda2v']) + 2*p_params['a3']*v


@njit
def omega_vy_der(v, p_params, rn_params):
    return (p_params['b'] + 0.5) * v**(p_params['b'] - 0.5)


@njit
def omega_vv_der(v, p_params, rn_params):
    return 2*p_params['b']*v**(2*p_params['b'] - 1)


@njit
def omega_v_til(v, p_params, rn_params):
    return omega_v(v, p_params, rn_params) - omega_v_der(v, p_params, rn_params)*v


@njit
def omega_vv_til(v, p_params, rn_params):
    return omega_vv(v, p_params, rn_params) - omega_vv_der(v, p_params, rn_params)*v


@njit
def omega_vy_til(v, p_params, rn_params):
    return omega_vy(v, p_params, rn_params) - omega_vy_der(v, p_params, rn_params)*v


@njit    
def A_2(v_st, p_params, rn_params):
    return 0.5*p_params['sigv']**2*omega_vv_der(v_st, p_params, rn_params)


@njit
def psi_A(v_st, p_params, rn_params, phi):
    return p_params['rho']*p_params['sigv']*omega_vy_der(v_st, p_params, rn_params)*phi*1j + omega_v_der(v_st, p_params, rn_params)


@njit
def A_0(v_st, p_params, rn_params, phi):
    return rn_params['lambda1y']*1j*phi - phi**2/2


@njit
def gamma_A(v_st, p_params, rn_params, phi):
    psi = psi_A(v_st, p_params, rn_params, phi)
    A2 = A_2(v_st, p_params, rn_params)
    return np.sqrt(psi**2 - 4*A2*A_0(v_st, p_params, rn_params, phi))


@njit
def varsigma_A(v_st, p_params, rn_params, phi):
    gamma = gamma_A(v_st, p_params, rn_params, phi)
    psi = psi_A(v_st, p_params, rn_params, phi)
    A2 = A_2(v_st, p_params, rn_params)
    return (gamma - psi)/(gamma + psi)


@njit
def A(tau, p_params, rn_params, phi, v_st):
    gamma = gamma_A(v_st, p_params, rn_params, phi)
    psi = psi_A(v_st, p_params, rn_params, phi)
    varsigma = varsigma_A(v_st, p_params, rn_params, phi)
    omega_vv_der_v_st = omega_vv_der(v_st, p_params, rn_params)
    return (-psi + gamma*(1-varsigma*np.exp(gamma*tau))/(1 + varsigma*np.exp(gamma*tau)))/(p_params['sigv']**2*omega_vv_der_v_st)


@njit
def int_A(tau, p_params, rn_params, phi, v_st):
    gamma = gamma_A(v_st, p_params, rn_params, phi)
    psi = psi_A(v_st, p_params, rn_params, phi)
    varsigma = varsigma_A(v_st, p_params, rn_params, phi)
    omega_vv_der_v_st = omega_vv_der(v_st, p_params, rn_params)
    return (2*np.log((varsigma +1)/(varsigma*np.exp(gamma*tau) + 1)) + tau*(gamma - psi))/p_params['sigv']**2/omega_vv_der_v_st


@njit
def int_A_2(tau, p_params, rn_params, phi, v_st):
    gamma = gamma_A(v_st, p_params, rn_params, phi)
    psi = psi_A(v_st, p_params, rn_params, phi)
    varsigma = varsigma_A(v_st, p_params, rn_params, phi)
    omega_vv_der_v_st = omega_vv_der(v_st, p_params, rn_params)
    num = 4*psi*np.log((varsigma*np.exp(gamma*tau) + 1)/(varsigma +1)) + tau*(gamma - psi)**2 +  4*gamma*(1/(varsigma*np.exp(gamma*tau)+1) - 1/(varsigma+1))
    return num/(p_params['sigv']**2*omega_vv_der_v_st)**2


@njit
def int_A_inv(tau, p_params, rn_params, phi, v_st):
    gamma = gamma_A(v_st, p_params, rn_params, phi)
    psi = psi_A(v_st, p_params, rn_params, phi)
    varsigma = varsigma_A(v_st, p_params, rn_params, phi)
    omega_vv_der_v_st = omega_vv_der(v_st, p_params, rn_params)
    D_pl = gamma + p_params['muv']*p_params['sigv']**2*omega_vv_der_v_st + psi
    D_mn = gamma - p_params['muv']*p_params['sigv']**2*omega_vv_der_v_st - psi
    num = p_params['sigv']**2*omega_vv_der_v_st*(2*np.log((D_mn - D_pl*varsigma*np.exp(gamma*tau))/(D_mn - D_pl*varsigma)) - tau*D_pl)
    return num/(D_pl*D_mn)


@njit
def B(tau, p_params, rn_params, phi, v_st):
    B_0 = p_params['lambda']*p_params['muv']*np.exp(1j*phi*(p_params['muy'] + p_params['lambda']) - p_params['sigy']**2*phi**2/2)
    B_1 = omega_v_til(v_st, p_params, rn_params)  + p_params['rho']*p_params['sigv']*omega_vy_til(v_st, p_params, rn_params)*1j*phi
    B_2 = 0.5*p_params['sigv']**2*omega_vv_til(v_st, p_params, rn_params)
    int_A_2_tau = int_A_2(tau, p_params, rn_params, phi, v_st)
    int_A_tau = int_A(tau, p_params, rn_params, phi, v_st)
    int_A_inv_tau = int_A_inv(tau, p_params, rn_params, phi, v_st)
    omega_vv_der_v_st = omega_vv_der(v_st, p_params, rn_params)
    return B_2*int_A_2_tau + B_1*int_A_tau + B_0*int_A_inv_tau + (p_params['mu'] + rn_params['lambda0y'])*1j*phi

