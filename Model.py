import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

class SystemModel:
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
    
    def next_step_nonlinear(self, theta, theta_dot, phi, phi_dot, tau, dt):
        x_plus_one = solve_ivp(
            lambda t, y: self.System_dynamics(y[0], y[1], y[2], y[3], tau), 
            [0, dt], 
            [theta, theta_dot, phi, phi_dot],
            method='RK45'
        )
        theta_x_plus_1, theta_dot_x_plus_1, phi_x_plus_1, phi_dot_x_plus_1 = x_plus_one.y[:, -1]
        return theta_x_plus_1, theta_dot_x_plus_1, phi_x_plus_1, phi_dot_x_plus_1
    
    def Linearised(self, theta, theta_dot, phi, phi_dot, tau, theta_eq, SamplingTime):
        ml = (self.mp * self.lp) + (self.mw * self.lw)
        u = tau
        '''
        x = np.array([theta_dot, theta, phi_dot, phi])
        A = np.array([
        [-self.b1/self.Jp, -(ml*self.g*np.cos(theta_eq))/self.Jp, self.b2/self.Jp, 0],
        [1, 0, 0, 0],
        [self.b1/self.Jp, (ml*self.g*np.cos(theta_eq))/self.Jp, -(self.b2/self.Jp + self.b2/self.Jw), 0],
        [0, 0, 1, 0]])
        B =  np.array([[-1],[0],[1],[0]])
        C = np.array([[0, 1, 0, 0],[0, 0, 0, 1]])
        '''
        # State vector: [theta, theta_dot, phi, phi_dot] imporant to keep the order consistent with the nonlinear model
        x = np.array([theta, theta_dot, phi, phi_dot])
        A = np.array([
            [0, 1, 0, 0], 
            [-(ml * self.g * np.cos(theta_eq))/self.Jp, -self.b1/self.Jp, 0, self.b2/self.Jp], 
            [0, 0, 0, 1], 
            [(ml * self.g * np.cos(theta_eq))/self.Jp,   self.b1/self.Jp, 0, -(self.b2/self.Jp + self.b2/self.Jw)]
        ])
        B =  np.array([0, -1, 0, 1])
        C = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        x_dot = A @ x + B * u
        y = C @ x   

        ## ZOH Discretization
        dim_x = A.shape[0]
        dim_y = C.shape[0]
        dim_u = B.shape[1]
        AB = np.zeros((self.dimx+self.dimu, self.dimx + self.dimu))
        AB[:self.dimx,:self.dimx] = self.A
        AB[:self.dimx,self.dimx:] = self.B
        exp = expm(AB*SamplingTime)
        Adisc = exp[:self.dimx, :self.dimx]
        Bdisc = exp[:self.dimx, self.dimx:]
        x_k_plus_1 = Adisc @ x + Bdisc * u

        return x_dot, y, x_k_plus_1, Adisc, Bdisc
