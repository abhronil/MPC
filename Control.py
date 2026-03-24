import numpy as np 
import matplotlib.pyplot as plt 
import cvxpy as cp
import control as ct
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from shapely.geometry import Polygon
import geopandas as gpd

class Controllers:
    def __init__(self, A, B, C, Q, R, N):
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
        self.Horizon = N
        self.dimx = A.shape[0]
        self.dimy = C.shape[0]
        self.dimu = B.shape[1]
        self.K, self.P, _ = ct.dlqr(A, B, Q, R)

    def forward(self, x, u):
        x_plus = self.A@x + self.B@u
        y = self.C@x
        return x_plus, y
    
    def ComputeXfineq(self, Ax, Au, bx, bu):
        """
        Input:
        Ax: Ineq matrix Ax <= bx
        Au: Ineq matrix Au <= bu
        bx: Bounds on x
        bu: Bounds on u
        Output:
        A_inf = ineq matrix Ax <= b for Xf
        b_inf = ineq vector Ax <= b for Xf
        """
        F = self.A-self.B@self.K
        Au_x = -Au@self.K
        constraintsA = np.vstack((Ax, Au_x))
        constraintsb = np.hstack((bx, bu))
        Ft = F.copy() 
        A_inf = constraintsA.copy()
        b_inf = constraintsb.copy()
        dim_con = Ax.shape[0]
        # Time step
        for t in range(100):
            # Propogate constraints to next state
            f = constraintsA @ Ft
            stopflag= True
            # Loop over all constraints
            for j in range(dim_con):
                # Maximize x such that the constraints are met this time step
                x = linprog(-f[j], A_ub=A_inf, b_ub=b_inf)["x"]
                #print(x)
                # Check if constraints are also met next time step, If yes then you are done
                if f[j]@x >= constraintsb[j]:
                    stopflag = False
                    break
            if stopflag:
                break
            # Propogate constraints to next state
            A_inf = np.vstack((A_inf, constraintsA @ Ft))
            b_inf = np.hstack((b_inf, constraintsb))
            # Propagate dynamics
            Ft = F @ Ft
        return A_inf, b_inf
    
    def ComputeXfellipse(self, Ax, Au, bx, bu, ax):
        
        Ellipsoid_m = self.P
        x = cp.Variable(self.dimx)
        cost = cp.quad_form(x, Ellipsoid_m)
        
        Au_x = -Au@self.K
        constraintsA = np.vstack((Ax, Au_x))
        constraintsb = np.hstack((bx, bu))
        con_size = constraintsb.size
        c_min = np.inf
        for i in range(con_size):
            constraint = [constraintsA[i]@x == constraintsb[i]]
            prob = cp.Problem(cp.Minimize(cost), constraint)
            prob.solve()
            
            c = prob.value
            if c < c_min:
                c_min = c
                
        P_inv = np.linalg.inv(Ellipsoid_m)
        # 2. Extract the 2x2 submatrix for the two states we want to draw
        # np.ix_ allows us to pull out the specific rows and columns safely
        c = c_min
        # Scale it by your constant c
        Cov = c * P_inv

        # 3. Generate 100 points around a standard unit circle
        angles = np.linspace(0, 2 * np.pi, 100)
        circle_points = np.vstack((np.cos(angles), np.sin(angles)))
        
        # 4. Transform the circle into our specific ellipse using Eigen decomposition
        # Cov = V * D * V^T, where V are eigenvectors and D are eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(Cov)

        # The transformation matrix is V * sqrt(D)
        transform_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        u1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])
        u2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1])
        vertex1 = u1+u2
        vertex2 = u1-u2
        vertex3 = -u1-u2
        vertex4 = -u1+u2
        # Apply transformation to the circle points
        ellipse_points = transform_matrix @ circle_points
        
        ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', lw=2, label=f'Vf(x) <= {c}')
        ax.fill(ellipse_points[0, :], ellipse_points[1, :], 'b', alpha=0.2)
        ax.plot(vertex1[0], vertex1[1], marker="o")
        ax.plot(vertex2[0], vertex2[1], marker="o")
        ax.plot(vertex3[0], vertex3[1], marker="o")
        ax.plot(vertex4[0], vertex4[1], marker="o")
        # Plot formatting
        ax.set_title(f"Terminal Region")
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.axis('equal') # Ensures the visual shape isn't distorted by axis stretching
        
        return 0
    
    def mpc(self, x0, TermSet = None, TermRHS = None):
        x = cp.Variable((self.dimx, self.Horizon+1))
        u = cp.Variable((self.dimu, self.Horizon))
        cost = 0
        constraints = []
        constraints += [x[:,0] == x0]
        for i in range(self.Horizon):
            # Dynamic constraint
            constraints += [x[:,i+1] == self.forward(x[:,i], u[:,i])[0]]
            
            # INPUT CONSTRAINTS
            constraints += [u[:,i] <= 0.5]
            constraints += [u[:,i]>= -0.5]

            # STATE CONSTRAINTS
            constraints += [x[0,i] <= 0.3]
            constraints += [x[0,i]>=-0.3]
            
            cost += 0.5*(cp.quad_form(x[:,i], self.Q) + cp.square(u[:,i])* self.R)
        
        #TERMINAL CONSTRAINT
        # IF Terminal constraint Set is not added in the MPC object, do not use it  
        if TermSet is not None and TermRHS is not None:   
            constraints += [TermSet@x[:,-1]<=TermRHS]
        
        #TERMINAL COST
        cost += 0.5*(cp.quad_form(x[:,-1], self.P))
        
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve()
        if problem.status != "optimal":
            print(f"Infeasible")
            return 0
        input = u.value[:,0]     
        return input
    
    # Add this to Dynamics Class
    def ZeroOrderHold(self, SamplingTime):
        # Trick from first tutorial in the MPC homework sessions 
        AB = np.zeros((self.dimx+self.dimu, self.dimx + self.dimu))
        AB[:self.dimx,:self.dimx] = self.A
        AB[:self.dimx,self.dimx:] = self.B
        exp = expm(AB*SamplingTime)
        
        Adisc = exp[:self.dimx, :self.dimx]
        Bdisc = exp[:self.dimx, self.dimx:]
        return Adisc, Bdisc
    
if __name__ == "__main__":
    # Setup matrices
    A = np.array([[100, 0],
                  [1, 2]])
    B = np.array([[1],
                  [0]])
    C = np.array([[-2/3,1]])
    x0 = np.array([3,3])
    Q = np.eye(2)
    R = np.array([1])

    # Time
    N = 5
    Time = 30
    t = np.arange(Time)
    # Create object
    my_sys = Controllers(A, B, C, Q, R, x0, N)
    # If we were to discretize the model:
    
    # Testing the Ellipse code
    #my_sys.ComputeXfellipse()
    
    """
    SamplingTime = 0.01
    Ad, Bd = my_sys.ZeroOrderHold(SamplingTime)
    print(f"Discreet A:\n{Ad}\n")
    print(f"Discreet B:\n{Bd}\n")
    # Placeholders for the trajectories
    x_hist = np.zeros((x0.shape[0], Time+1))
    u_hist = np.zeros((B.shape[1], Time))
    y_hist = np.zeros((C.shape[0], Time))
    x_hist[:,0] = x0
    # Run the simulation for "Time" timesteps
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
    
    # Code for testing terminal set
    """
    A1 = np.array([[2,1],
              [0,2]])
    B1 = np.eye(2)
    alpha = 1e-5
    Q1 = alpha*np.eye(2)
    R1 = np.eye(2)
    TermTest = Controllers(A1, B1, C, Q1, R1, x0, N)

    # Constraints A matrices
    # Input
    A_u = np.kron(np.eye(2), 
                  np.array([[1],
                            [-1]]))
    # State
    A_x = np.array([[1,0],
                    [-1,0]])
    # Constraint b matrices 
    b_u = 5*np.ones((4))
    b_x = 1*np.array([5,5])
    
    A_inf, b_inf = TermTest.ComputeXfineq(A_x, A_u, b_x, b_u)
    
    def plot_polygon(A, b, ax):
        '''
        Visualize the polytope defined by A x <= b.
        '''
        halfspaces = np.hstack((A, -b[:, np.newaxis]))
        feasible_point = np.zeros(A.shape[1])
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        polygon = Polygon(hs.intersections).convex_hull
        polygon_gpd = gpd.GeoSeries(polygon)
        polygon_gpd.plot(ax=ax, alpha=0.3)
        plt.plot(*polygon.exterior.xy, 'ro')
        plt.axis('equal')
        plt.grid()
        
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_polygon(A_inf, b_inf, ax)
    TermTest.ComputeXfellipse(A_x, A_u, b_x, b_u, ax)
    plt.show()
    #print(A_inf)
    #print(b_inf)
    

        