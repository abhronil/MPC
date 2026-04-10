import numpy as np
import matplotlib.pyplot as plt
from Model import SystemModel 
from Control import Controllers
import quadprog
import sys

params = {
    'Jw': 0.0025, 'Jp': 0.05, 
    'mp': 0.4, 'mw': 0.2, 
    'lp': 0.3, 'lw': 0.3, 
    'b1': 0.01, 'b2': 0.005,
    'SamplingTime': 0.1, 'Theta_eq': np.pi
}

plant = SystemModel(params)
A, B, C = plant.Linearised(params['Theta_eq'])
A_d, B_d = plant.ZeroOrderHold(params['SamplingTime'])


Q_val = 1e-4*np.array([100,1,0.1,0.001])
Q = np.diag(Q_val)
R = 1
yref = np.array([0,0])

total_time = 10.0
num_steps_dis = int(total_time / params["SamplingTime"])
t_d = np.arange(num_steps_dis + 1) * params['SamplingTime']

deviation = np.array([0.15, -0.1, 0.3, 1])
if params['Theta_eq'] == 0:
    x_eq = np.array([0., 0., 0., 0.])
else:
    x_eq = np.array([np.pi, 0., 0., 0.])

# Constraint matrices
Ax = np.array([
    [ 1,  0,  0,  0],  
    [-1,  0,  0,  0],  
    [ 0,  1,  0,  0],  
    [ 0, -1,  0,  0],  
    [ 0,  0,  1,  0],  
    [ 0,  0, -1,  0],  
    [ 0,  0,  0,  1],  
    [ 0,  0,  0, -1]   
])
gx = 0.3 * np.ones((2))
gx = np.hstack((gx, 100*np.ones(6)))

Au = np.zeros((2,1))
Au[0,0] = 1
Au[1,0] = -1 
gu = 0.5 * np.ones((2))

horizons = [5, 10, 20] # List of N values to test
results = {} # Dictionary to store trajectories for each N


for N in horizons:
    print(f"Simulating for N = {N}...")
    
    # Initialize Controller for current N
    Controller = Controllers(A_d, B_d, C, Q, R, N, yref)
    A_con, g_con = Controller.ComputeXfineq(Ax, Au, gx, gu)
    # P, gamma = Controller.computeXn(Ax, Au, gx, gu)

    # # Calculate worst positive sum of states still in the region of attraction 
    # deviation_max = Controller.Calculate_worst_state(P, gamma)
    deviation_maxTerm = Controller.Calculate_worst_state(A_con, g_con)
   # print(f"Maximum allowed deviation on the pendulum angle: {deviation_max}")
    print(f"Border of Xf: {deviation_maxTerm}")
    
    
    Check_ineq = A_con @ deviation
    if np.all(Check_ineq <= g_con):
        print("You are inside the terminal constraint set")
    else: 
        print("MPC will do something else")
        
    # Check_ineq2 = P @ deviation
    # if np.all(Check_ineq2 <= gamma):
    #     print("You are inside the RoA")
    # else: 
    #     print("MPC fails")
    # Initialize storage arrays
    x_lin_err = np.zeros((4, num_steps_dis + 1))
    u_lin = np.zeros(num_steps_dis)
    
    x_lin_err[:, 0] = deviation
    
    for k in range(num_steps_dis):
        tau_lin, _, _ = Controller.mpc(x_lin_err[:,k], Ax, gx, Au, gu, A_con, g_con)
        tau_lin = tau_lin[0]
        u_lin[k] = tau_lin
        
        tau_lin_vec = np.array([[tau_lin]])
        x_lin_err[:,[k+1]], _ = plant.forward_discreet_linear(x_lin_err[:,[k]], tau_lin_vec)
        
    results[N] = {
        'x': x_lin_err + x_eq[:, np.newaxis],
        'u': u_lin
    }



labels = ['θ (rad)', 'θ̇ (rad/s)', 'φ (rad)', 'φ̇ (rad/s)']
titles = ['Pendulum angle', 'Pendulum angular velocity', 'Wheel angle', 'Wheel angular velocity']

fig, axes = plt.subplots(3, 1, figsize=(6, 12))
fig.suptitle('MPC Horizon (N) Comparison')

colors = ['tab:blue', 'tab:orange', 'tab:green']

ax_wheel = axes[0]
for idx, N in enumerate(horizons):
    ax_wheel.stairs(results[N]['x'][2, :-1], t_d, color=colors[idx % len(colors)], label=f'N={N}', linewidth=1.5)
ax_wheel.set_ylabel('φ (rad)')
ax_wheel.grid(True)
ax_wheel.set_xlim([0,10])
ax_wheel.legend()

ax_pend = axes[1]
for idx, N in enumerate(horizons):
    ax_pend.stairs(results[N]['x'][0, :-1], t_d, color=colors[idx % len(colors)], label=f'N={N}', linewidth=1.5)
ax_pend.set_ylabel('θ (rad)')
#ax_pend.set_title('Pendulum angle')
ax_pend.grid(True)
ax_pend.legend()

ax_pend.set_ylim([x_eq[0] - 0.3, x_eq[0] + 0.3])
ax_pend.set_xlim([0,10])


ax_u = axes[2]
for idx, N in enumerate(horizons):
    ax_u.stairs(results[N]['u'], t_d, color=colors[idx % len(colors)], label=f'N={N}', linewidth=1.5)
ax_u.set_ylabel('τ (N·m)')
#ax_u.set_title('Control torque')
ax_u.set_xlabel('Time (s)')
ax_u.legend()
ax_u.set_xlim([0,10])
ax_u.grid(True)

plt.tight_layout()
plt.show()