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
        self.T = params["SamplingTime"]
        self.theta_eq = params['Theta_eq']
        self.g = 9.81
        self.A, self.B, self.C = self.Linearised(theta_eq=self.theta_eq)
        self.dimx = self.A.shape[0]
        self.dimy = self.C.shape[0]
        self.dimu = self.B.shape[1]
        self.A_d, self.B_d = self.ZeroOrderHold(SamplingTime=self.T)


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
    
    def forward_discreet_linear(self, x, u):
        x_plus = self.A_d@x + self.B_d@u
        y = self.C@x
        return x_plus, y
    def Linearised(self, theta_eq):
        self.ml = (self.mp * self.lp) + (self.mw * self.lw)
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
        # x = np.array([theta, theta_dot, phi, phi_dot])
        A = np.array([
            [0, 1, 0, 0], 
            [-(self.ml * self.g * np.cos(theta_eq))/self.Jp, -self.b1/self.Jp, 0, self.b2/self.Jp], 
            [0, 0, 0, 1], 
            [(self.ml * self.g * np.cos(theta_eq))/self.Jp,   self.b1/self.Jp, 0, -(self.b2/self.Jp + self.b2/self.Jw)]
        ])
        B =  np.array([[0], 
                       [-1/self.Jp], 
                       [0], 
                       [1/self.Jw + 1/self.Jp]])
        
        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        # Put this into another continuous forward function
        # x_dot = A @ x + B * u
        # y = C @ x   
        return A, B, C
    
    def ZeroOrderHold(self, SamplingTime):
        # Trick from first tutorial in the MPC homework sessions 
        AB = np.zeros((self.dimx+self.dimu, self.dimx + self.dimu))
        AB[:self.dimx,:self.dimx] = self.A
        AB[:self.dimx,self.dimx:] = self.B
        exp = expm(AB*SamplingTime)
        
        Adisc = exp[:self.dimx, :self.dimx]
        Bdisc = exp[:self.dimx, self.dimx:]
        return Adisc, Bdisc
