import numpy as np
from scipy.integrate import solve_ivp

class nonLinearModel:
    def __init__(self, params):
        self.Jw = params['Jw']
        self.Jp = params['Jp']
        self.mp = params['mp']
        self.mw = params['mw']
        self.lp = params['lp']
        self.lw = params['lw']
        self.b1 = params['b1']
        self.b2 = params['b2']
        self.g = 9.81

    def System_dynamics(self, theta, theta_dot, phi, phi_dot, tau):
        ml = (self.mp * self.lp) + (self.mw * self.lw)
        theta_dot_dot = (-tau - ml * self.g * np.sin(theta) - (self.b1 * theta_dot) + (self.b2 * phi_dot)) / self.Jp
        phi_dot_dot = ((tau - self.b2 * phi_dot) / self.Jw) - theta_dot_dot
        return [theta_dot, theta_dot_dot, phi_dot, phi_dot_dot]
    
    def next_step(self, theta, theta_dot, phi, phi_dot, tau, dt):
        x_plus_one = solve_ivp(
            lambda t, y: self.System_dynamics(y[0], y[1], y[2], y[3], tau), 
            [0, dt], 
            [theta, theta_dot, phi, phi_dot],
            method='RK45'
        )
        theta_x_plus_1, theta_dot_x_plus_1, phi_x_plus_1, phi_dot_x_plus_1 = x_plus_one.y[:, -1]
        return theta_x_plus_1, theta_dot_x_plus_1, phi_x_plus_1, phi_dot_x_plus_1