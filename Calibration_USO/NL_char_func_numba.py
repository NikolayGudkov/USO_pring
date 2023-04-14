import numpy as np
import numba
from numba import jit, float64


@jit(float64(float64, numba.float64[:]))
def omega_v(v,p_params, rn_params):
    return (params['alp_0'] + params['lambda_0_v']) + params['alp_1']/v + (params['alp_2'] + params['lambda_1_v'])*v + params['alp_3']*v**2

def omega_vy(v, params):
    return v**(params['b'] + 0.5)

def omega_vv(v, params):
    return v**(2 * params['b'] )

def omega_v_der(v, params):
    return -params['alp_1']/v**2 + (params['alp_2'] + params['lambda_1_v']) + 2*params['alp_3']*v

def omega_vy_der(v, params):
    return (params['b'] + 0.5) * v**(params['b'] - 0.5)

def omega_vv_der(v, params):
    return 2*params['b']*v**(2*params['b'] - 1)

def omega_v_til(v, params):
    return omega_v(v, params) - omega_v_der(v, params)*v

def omega_vv_til(v, params):
    return omega_vv(v, params) - omega_vv_der(v, params)*v

def omega_vy_til(v, params):
    return omega_vy(v, params) - omega_vy_der(v, params)*v
    
def A_2(v_st, params):
    return 0.5*params['sigma_v']**2*omega_vv_der(v_st, params)

def psi_A(v_st, params, phi):
    return params['rho']*params['sigma_v']*omega_vy_der(v_st, params)*phi*1j + omega_v_der(v_st, params)

def A_0(v_st, params, phi):
    return params['lambda_1_y']*1j*phi - phi**2/2

def gamma_A(v_st, params, phi):
    return np.sqrt(psi_A(v_st, params, phi)**2 - 4*A_2(v_st, params)*A_0(v_st, params, phi))

def varsigma_A(v_st, params, phi):
    gamma = gamma_A(v_st, params, phi)
    psi = psi_A(v_st, params, phi)
    return (gamma - psi)/(gamma + psi)

def A(tau, params, phi, v_st):
    gamma = gamma_A(v_st, params, phi)
    psi = psi_A(v_st, params, phi)
    varsigma = varsigma_A(v_st, params, phi)
    return (-psi + gamma*(1-varsigma*np.exp(gamma*tau))/(1 + varsigma*np.exp(gamma*tau)))/(params['sigma_v']**2*omega_vv_der(v_st, params))

def int_A(tau, params, phi, v_st):
    gamma = gamma_A(v_st, params, phi)
    psi = psi_A(v_st, params, phi)
    varsigma = varsigma_A(v_st, params, phi)
    return (2*np.log((varsigma +1)/(varsigma*np.exp(gamma*tau) + 1)) + tau*(gamma - psi))/params['sigma_v']**2/omega_vv_der(v_st, params)

def int_A_2(tau, params, phi, v_st):
    gamma = gamma_A(v_st, params, phi)
    psi = psi_A(v_st, params, phi)
    varsigma = varsigma_A(v_st, params, phi)
    num = 4*psi*np.log((varsigma*np.exp(gamma*tau) + 1)/(varsigma +1)) + tau*(gamma - psi)**2 +  4*gamma*(1/(varsigma*np.exp(gamma*tau)+1) - 1/(varsigma+1))
    return num/(params['sigma_v']**2*omega_vv_der(v_st, params))**2

def int_A_inv(tau, params, phi, v_st):
    gamma = gamma_A(v_st, params, phi)
    psi = psi_A(v_st, params, phi)
    varsigma = varsigma_A(v_st, params, phi)
   
    D_pl = gamma + params['mu_v']*params['sigma_v']**2*omega_vv_der(v_st, params) + psi
    D_mn = gamma - params['mu_v']*params['sigma_v']**2*omega_vv_der(v_st, params) - psi
    
    num = params['sigma_v']**2*omega_vv_der(v_st, params)*(2*np.log((D_mn - D_pl*varsigma*np.exp(gamma*tau))/(D_mn - D_pl*varsigma)) - tau*D_pl)
    return num/(D_pl*D_mn)

def B(tau, params, phi, v_st):
    B_0 = params['lambda']*params['mu_v']*np.exp(1j*phi*(params['mu_y'] + params['lambda_j']) - params['sigma_y']**2*phi**2/2)
    B_1 = omega_v_til(v_st, params)  + params['rho']*params['sigma_v']*omega_vy_til(v_st, params)*1j*phi
    B_2 = 0.5*params['sigma_v']**2*omega_vv_til(v_st, params)
    
    return B_2*int_A_2(tau, params, phi, v_st) + B_1*int_A(tau, params, phi, v_st) + B_0*int_A_inv(tau, params, phi, v_st) + (params['mu'] + params['lambda_0_y'])*1j*phi
   
def char_fun(phi, y, v, t, s, v_st, params):
    tau = s - t
    return np.exp(1j*phi*y + v*A(tau, params, phi, v_st) + B(tau, params, phi, v_st))


