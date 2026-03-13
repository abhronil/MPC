import numpy as np 
import matplotlib.pyplot as plt 
import cvxpy as cp
import control as ct

class MPC:
    def __init__(self, A, B, C, Q, R, x0, N):
        """
        Inputs: 
        A - State dynamics, (n X n)
        B - Input matrix, (n X m)
        C - Output matrix (p X n)
        Q - State cost (n X n)
        R - Input cost (m X m)
        x0 - Initial state for the MPC problem (n X 1)
        N - Horizon length
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.Horizon = N
        self.dimx = A.shape[0]
        self.dimy = C.shape[0]
        self.dimu = B.shape[1]
        _, self.P, _ = ct.dlqr(A, B, Q, R)

    def forward(self, x, u):
        x_plus = self.A@x + self.B@u
        y = self.C@x
        return x_plus, y
    
    def mpc(self,x0):
        x = cp.Variable((self.dimx, self.Horizon+1))
        u = cp.Variable((self.dimu, self.Horizon))
        cost = 0
        constraints = []
        constraints += [x[:,0] == x0]
        for i in range(self.Horizon):
            # Dynamic constraint
            constraints += [x[:,i+1] == self.forward(x[:,i], u[:,i])[0]]
            
            # INPUT CONSTRAINTS
            #constraints += [u[:,i] <= 1]
            #constraints += [u[:,i]>= -1]

            # STATE CONSTRAINTS
            #constraints += [x[0,i] <= 5]
            #constraints += [x[0,i]>=-5]
            
            
            cost += 0.5*(cp.quad_form(x[:,i], self.Q) + cp.square(u[:,i])@ self.R)
        
        #TERMINAL CONSTRAINT    
        #constraints += [x[:,-1]==0]
        
        #TERMINAL COST
        #cost += 0.5*(cp.quad_form(x[:,-1], self.P) )
        
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve()
        print(f"MPC solved with status: {problem.status}")
        input = u.value[:,0]     
        return input
    
if __name__ == "__main__":
    A = np.array([[4/3, -2/3],
                  [1, 0]])

    B = np.array([[1],
                  [0]])
    
    C = np.array([[-2/3,1]])

    x0 = np.array([3,3])

    Q = np.eye(2)

    R = np.array([1])

    N = 5
    Time = 50
    t = np.arange(Time)

    my_sys = MPC(A, B, C, Q, R, x0, N)

    x_hist = np.zeros((x0.shape[0], Time+1))
    u_hist = np.zeros((B.shape[1], Time))
    y_hist = np.zeros((C.shape[0], Time))
    x_hist[:,0] = x0
    
    for i in range(Time):
        u = my_sys.mpc(x_hist[:,i])
        x_plus, y = my_sys.forward(x_hist[:,i], u)
        
        u_hist[:,i] = u 
        x_hist[:,i+1] = x_plus
        y_hist[:,i] = y
    fig, ax = plt.subplots(2,1, figsize=(16,8))

    ax[0].plot(t, y_hist[0], label=f'$y$')
    ax[0].set_ylabel("y(k)")
    ax[1].plot(t, u_hist[0], label=f'$u$')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("u(k)")
    ax[0].set_title("MPC, No constraints")
    ax[0].grid()
    ax[1].grid()


    plt.show()

        