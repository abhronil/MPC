import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Model import SystemModel 
from Control import Controllers
import control as ct
import quadprog
import sys
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Subplot title size
    'axes.labelsize': 20,     # X/Y axis label size
    'xtick.labelsize': 12,    # X axis tick labels (numbers)
    'ytick.labelsize': 12,    # Y axis tick labels (numbers)
    'legend.fontsize': 16,    # Legend text size
    'figure.titlesize': 25    # Main figure title size
})
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
N = 5


Q_vallqr = 1e-8*np.array([100,1,0.1,0.001])
Q_LQR = np.diag(Q_vallqr)
R_LQR = R
yref = np.array([0,0])
Controller = Controllers(A_d, B_d, C, Q, R, N, yref)
Controller_LQR = Controllers(A_d, B_d, C, Q_LQR, R_LQR, N, yref)

omega_n = np.sqrt(plant.Jp / (plant.ml * plant.g))
Sampling_time = 0.2/omega_n

Controllability = ct.ctrb(A, B)
if np.linalg.matrix_rank(Controllability) == 4:
    print("System is controllable!")
else:  
    print(f"Boohoo")


dt = 0.01
total_time = 10.0
num_steps = int(total_time / dt)
num_steps_dis = int(total_time / params["SamplingTime"])
x_lin_err = np.zeros((4,num_steps_dis+1))
y_lin_err = np.zeros((2,num_steps_dis))
u_nl = []
u_lin = []
# LQR
x_LQR = np.zeros((4,num_steps_dis+1))
y_LQR = np.zeros((2,num_steps_dis))
u_LQR = []

x_nl = np.zeros((4, num_steps + 1))

VF_diff = np.zeros(num_steps_dis)
stage_cost = np.zeros(num_steps_dis)


calc_u_count = int(params["SamplingTime"] / dt)
# Constraint matrices for Xf 

# Add large constraints on all the states just so the compute Xn can find a region always
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
# Ax = np.array([
#     [ 1,  0,  0,  0],  
#     [-1,  0,  0,  0],  
# ])
gx = 0.3 * np.ones((2))
gx = np.hstack((gx, 100*np.ones(6)))


Au = np.zeros((2,1))
Au[0,0] = 1
Au[1,0] = -1 
gu= 0.5 * np.ones((2))

A_con, g_con = Controller.ComputeXfineq(Ax, Au, gx, gu)


P, gamma = Controller.computeXn(Ax, Au, gx, gu)

# Calculate worst positive sum of states still in the region of attraction 
deviation_max = Controller.Calculate_worst_state(P, gamma)
deviation_max_A = Controller.Calculate_worst_state(A_con, g_con)
print(f"Maximum allowed deviation on the pendulum angle: {deviation_max}")
print(f"Maximum allowed deviation on the pendulum angle 2: {deviation_max_A}")
deviation = np.array([0.15, -0.1, 0.3, 1])
#deviation = np.array([0.103,0,0,0])
try:
    Check_ineq = P @ deviation
    if not np.all(Check_ineq <= gamma):
        raise ValueError(f"Deviation {deviation} is outside the Region of Attraction!")
    print("State is safe. Proceeding with MPC...")
except ValueError as e:
    print(f"Deviation is too large for feasible solution")
    print(f"Setting the maximum deviation as the initial state")
    deviation = deviation_max

if params['Theta_eq'] == 0:
    x_eq = np.array([0., 0., 0., 0.])
    x0 = x_eq+deviation
else:
    x_eq = np.array([np.pi, 0., 0., 0.])
    x0 = x_eq+deviation
    
x_nl[:, 0] = x0
x_lin_err[:, 0] = deviation
x_LQR[:,0] = deviation

theta, theta_dot, phi, phi_dot = x0[0], x0[1], x0[2], x0[3]
# if np.all(Check_ineq <= gamma):
#     print("Control admissible sequence is found")
# else: 
#     print("bing bong")
    

for i in range(num_steps):
    if i % calc_u_count == 0:
        k = i // calc_u_count
        error_nl = x_nl[:, i] - x_eq
        
        tau_lin, VF_diff[k], stage_cost[k] = Controller.mpc(x_lin_err[:,k],Ax, gx, Au, gu, A_con, g_con)
        
        tau_LQR = Controller_LQR.lqr(x_LQR[:,k])
        tau_LQR = tau_LQR[0]
        tau_LQR_Vec = np.array([[tau_LQR]])
        u_LQR.append(tau_LQR)
        
        tau_lin = tau_lin[0]
        tau_nl,_,_ = Controller.mpc(error_nl,Ax, gx, Au, gu, A_con, g_con)
        tau_nl = tau_nl[0]
        tau_lin_vec = np.array([[tau_lin]])
        u_nl.append(tau_nl)
        u_lin.append(tau_lin)
        
        x_lin_err[:,[k+1]], y_lin_err[:,[k]] = plant.forward_discreet_linear(x_lin_err[:,[k]], tau_lin_vec)
        
        x_LQR[:,[k+1]], y_LQR[:,[k]] = plant.forward_discreet_linear(x_LQR[:,[k]], tau_LQR_Vec)
        # Stability Assumptions
        # Vf_future = Controller.CalcTerminalCost(x_lin_err[:,[k+1]])
        # Vf_now = Controller.CalcTerminalCost(x_lin_err[:,[k]])
        # stageC = Controller.CalcStageCost(x_lin_err[:,[k]], tau_lin)
        # Calculate this IN THE MPC LOOP. IT HAS TO BE THE FINAL STATE IN THE HORIZON STUPID 
        
        
        
    # Non linear Model
    theta, theta_dot, phi, phi_dot = plant.next_step_nonlinear(theta, theta_dot, phi, phi_dot, tau_nl, dt)
    x_nl[:, i+1] = [theta, theta_dot, phi, phi_dot]
u_nl = np.array(u_nl)
u_lin = np.array(u_lin)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(True)


L = params['lp'] + params['lw']
ax.set_xlim(-L - 0.2, L + 0.2)
ax.set_ylim(-L - 0.2, L + 0.2)
ax.set_title("Reaction Wheel Pendulum - Falling from Top")


pendulum_arm, = ax.plot([], [], 'b-', lw=4, label='Pendulum Arm')
wheel_radius = 0.1
wheel_circle = plt.Circle((0, 0), wheel_radius, color='g', fill=False, lw=2)
ax.add_patch(wheel_circle)
wheel_spoke, = ax.plot([], [], 'r-', lw=2)

def update(frame):
    current_theta = x_nl[0,frame]
    current_phi = x_nl[2,frame]
    wheel_x = L * np.sin(current_theta)
    wheel_y = -L * np.cos(current_theta) 
    pendulum_arm.set_data([0, wheel_x], [0, wheel_y])
    wheel_circle.center = (wheel_x, wheel_y)
    absolute_wheel_angle = current_theta + current_phi
    spoke_end_x = wheel_x + wheel_radius * np.sin(absolute_wheel_angle)
    spoke_end_y = wheel_y - wheel_radius * np.cos(absolute_wheel_angle)
    wheel_spoke.set_data([wheel_x, spoke_end_x], [wheel_y, spoke_end_y])
    return pendulum_arm, wheel_circle, wheel_spoke
ani = animation.FuncAnimation(
    fig, update, frames=num_steps, interval=dt * 1000, blit=True
)

plt.show()


t = np.arange(num_steps + 1) * dt
t_d = np.arange(num_steps_dis+1) * params['SamplingTime']
labels = ['θ (rad)', 'θ̇ (rad/s)', 'φ (rad)', 'φ̇ (rad/s)']
titles = ['Pendulum angle', 'Pendulum angular velocity', 'Wheel angle', 'Wheel angular velocity']
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle('Linear MPC vs LQR Response')

# 1. Pendulum angle (State index 0)
axes[0].stairs(x_lin_err[0, :-1] + x_eq[0], t_d, color='tab:blue', label='Linear MPC', linewidth=2)
axes[0].stairs(x_LQR[0, :-1] + x_eq[0], t_d, color='tab:green', label='LQR', linewidth=2, linestyle='--')
axes[0].set_ylabel('θ (rad)')

axes[0].set_ylim([x_eq[0] - 0.3, x_eq[0] + 0.3])
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_xlim([0,10])

# 2. Wheel angle (State index 2)
axes[1].stairs(x_lin_err[2, :-1] + x_eq[2], t_d, color='tab:blue', label='Linear MPC', linewidth=2)
axes[1].stairs(x_LQR[2, :-1] + x_eq[2], t_d, color='tab:green', label='LQR', linewidth=2, linestyle='--')
axes[1].set_ylabel('φ (rad)')

axes[1].set_xlim([0,10])
axes[1].legend(loc='upper right')
axes[1].grid(True)

# 3. Control torque
axes[2].stairs(u_lin, t_d, color='tab:blue', label='Torque MPC', linewidth=2)
axes[2].stairs(u_LQR, t_d, color='tab:green', label='Torque LQR', linewidth=2, linestyle='--')
axes[2].set_ylabel('τ (N·m)')

axes[2].set_xlabel('Time (s)')
axes[2].set_xlim([0,10])
axes[2].legend(loc='upper right')
axes[2].grid(True)

plt.tight_layout()
plt.show()