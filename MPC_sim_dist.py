import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Model import SystemModel 
from Control import Controllers
import control as ct

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
N = 1
yref = np.array([0])
Controller = Controllers(A_d, B_d, C, Q, R, N, yref)

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

x_obs = np.zeros((4,num_steps_dis+1))
y_obs = np.zeros((2,num_steps_dis))

x_obs_nl = np.zeros((4,num_steps_dis+1))
y_obs_nl = np.zeros((2,num_steps_dis))

d_lin = np.zeros((1,num_steps_dis+1))
d_nl = np.zeros((1,num_steps_dis+1))
u_nl = []
u_lin = []

x_nl = np.zeros((4, num_steps + 1))
deviation = np.array([0.0, 0, -0.0, 0.0])


calc_u_count = int(params["SamplingTime"] / dt)
# Constraint matrices for Xf 
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
gx = np.hstack((gx, 1000*np.ones(6)))


Au = np.zeros((2,1))
Au[0,0] = 1
Au[1,0] = -1 
gu= 0.5 * np.ones((2))

A_con, g_con = Controller.ComputeXfineq(Ax, Au, gx, gu)

P, gamma = Controller.computeXn(Ax, Au, gx, gu)

deviation_max = Controller.Calculate_worst_state(P, gamma)
print(f"Maximum allowed deviation on the pendulum angle: {deviation_max}")
deviation = np.array([0.0,0,0,0])

try:
    Check_ineq = P @ deviation
    if not np.all(Check_ineq <= gamma):
        raise ValueError(f"Deviation {deviation} is outside the Region of Attraction!")
    print("State is safe. Proceeding with MPC...")
except ValueError as e:
    print(f"Deviation is too large for feasible solution")
    print(f"Setting the maximum deviation as the initial state")
    deviation = deviation_max
    
d = np.array([[-0.1]])
if params['Theta_eq'] == 0:
    x_eq = np.array([0., 0., 0., 0.])
    x0 = x_eq+deviation
else: 
    x_eq = np.array([np.pi, 0., 0., 0.])
    x0 = x_eq+deviation
    
Initial_deviation_err = np.array([-0.05, -0.010, 0,0.01])

x_nl[:, 0] = x0
x_lin_err[:, 0] = deviation
x_obs[:,0] = deviation + Initial_deviation_err
x_obs_nl[:,0] = deviation + Initial_deviation_err

theta, theta_dot, phi, phi_dot = x0[0], x0[1], x0[2], x0[3]
#print(x_target, u_target)
y_nl = np.array([[deviation[0]],[deviation[2]]])
for i in range(num_steps):
    if i % calc_u_count == 0:
        k = i // calc_u_count
        if k == num_steps_dis//2:
            d = np.array([[0.0]])
        error_nl = x_obs_nl[:, k]
        tau_nl,_,_ = Controller.mpc(error_nl,Ax, gx, Au, gu, A_con, g_con, dist=d_nl[:,[k]])  
        tau_nl = tau_nl[0]
        tau_lin,_,_ = Controller.mpc(x_obs[:,k],Ax, gx, Au, gu, A_con, g_con, dist=d_lin[:,[k]])
        tau_lin = tau_lin[0]
        tau_lin_vec = np.array([[tau_lin]])
        u_nl.append(tau_nl)
        u_lin.append(tau_lin)
        # lin
        x_aug = np.vstack((x_obs[:,[k]], d_lin[:,[k]]))
        x_lin_err[:,[k+1]], y_lin_err[:,[k]] = Controller.forward_real(x_lin_err[:,[k]], tau_lin_vec, d)
        x_obs[:,[k+1]], d_lin[:,[k+1]], y_obs[:,[k]] = Controller.observ_forward(x_aug, y_lin_err[:,[k]], tau_lin_vec)
        # NL
        x_aug_nl = np.vstack((x_obs_nl[:,[k]], d_nl[:,[k]]))
        x_obs_nl[:,[k+1]], d_nl[:,[k+1]], y_obs_nl[:,[k]] = Controller.observ_forward(x_aug_nl, y_nl, np.array([[tau_nl]]))
        
    theta, theta_dot, phi, phi_dot = plant.next_step_nonlinear(theta, theta_dot, phi, phi_dot, tau_nl, dt, d[0,0])
    x_nl[:, i+1] = [theta, theta_dot, phi, phi_dot]
    y_nl = np.array([[theta],[phi]]) - np.array([[np.pi],[0]]) + np.random.multivariate_normal(np.zeros(2), 1e-6*np.eye(2)).reshape(-1,1)

x_lin_err[0] += np.pi
x_obs[0] += np.pi
x_obs_nl[0]+=np.pi
t = np.arange(num_steps + 1) * dt
t_d = np.arange(num_steps_dis + 1) * params['SamplingTime']
u_lin = np.array(u_lin)

fig, axes = plt.subplots(4, 1, figsize=(12, 24))
fig.suptitle('Observer-based MPC Response')

# --- Pendulum angle error ---
#axes[0].stairs(x_lin_err[0, :-1], t_d, label='True State', linewidth=2)
axes[0].plot(t, x_nl[0], label='True State', linewidth=2)
#axes[0].stairs(x_obs[0, :-1], t_d, linestyle='--', label='Observed', linewidth=2)
axes[0].set_ylabel('θ error (rad)')

axes[0].set_ylim([np.pi - 0.3, np.pi + 0.3]) 
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_xlim([0,10])

# --- Wheel angle error ---
#axes[1].stairs(x_lin_err[2, :-1], t_d, label='True State', linewidth=2)
#axes[1].stairs(x_obs[2, :-1], t_d, linestyle='--', label='Observed', linewidth=2)
axes[1].plot(t, x_nl[2], label='True State', linewidth=2)
axes[1].set_ylabel('φ error (rad)')
axes[1].legend(loc='upper right')
axes[1].grid(True)
axes[1].set_xlim([0,10])
# --- Disturbance estimate ---
axes[2].stairs(d_nl[0, :-1], t_d, color='tab:orange', label='Estimated disturbance', linewidth=2)
axes[2].axhline(d[0, 0], color='k', linestyle='--', label=f'True dist 2 ({d[0,0]})', linewidth=2)
axes[2].axhline(-0.1, color='k', linestyle='--', label=f'True dist 1 ({-0.1})', linewidth=2)
axes[2].set_ylabel('d')

axes[2].legend(loc='upper right')
axes[2].grid(True)
axes[2].set_xlim([0,10])

# --- Control torque ---
axes[3].stairs(u_nl, t_d, color='tab:red', label='Control torque', linewidth=2)
axes[3].set_ylabel('τ (N·m)')

axes[3].set_xlabel('Time (s)')
axes[3].legend(loc='upper right')
axes[3].grid(True)
axes[3].set_xlim([0,10])

plt.tight_layout(pad=5)
plt.show()




"""
for i in range(num_steps):
    if i % calc_u_count == 0:
        k = i // calc_u_count
        error_nl = x_nl[:, i] - x_eq
        tau_nl = Controller.mpc(error_nl, A_con, g_con)[0]
        tau_lin = Controller.mpc(x_lin_err[:,k], A_con, g_con)[0]
        tau_lin_vec = np.array([[tau_lin]])
        u_nl.append(tau_nl)
        u_lin.append(tau_lin)
        
        x_lin_err[:,[k+1]], y_lin_err[:,[k]] = plant.forward_discreet_linear(x_lin_err[:,[k]], tau_lin_vec)
    
    # Non linear Model
    theta, theta_dot, phi, phi_dot = plant.next_step_nonlinear(theta, theta_dot, phi, phi_dot, tau_nl, dt)
    x_nl[:, i+1] = [theta, theta_dot, phi, phi_dot]
u_nl = np.array(u_nl)
u_lin = np.array(u_lin)
"""
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(False)


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

"""
t = np.arange(num_steps + 1) * dt
t_d = np.arange(num_steps_dis+1) * params['SamplingTime']
labels = ['θ (rad)', 'θ̇ (rad/s)', 'φ (rad)', 'φ̇ (rad/s)']
titles = ['Pendulum angle', 'Pendulum angular velocity', 'Wheel angle', 'Wheel angular velocity']
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
fig.suptitle('Linear vs Nonlinear - MPC Response')

for i, ax in enumerate(axes.flat[:4]):
    ax.plot(t, x_nl[i], linestyle='--', label='Nonlinear')
    ax.stairs(x_lin_err[i,:-1] + x_eq[i],t_d, linestyle='-', label='Linear')
    ax.set_ylabel(labels[i])
    ax.set_title(titles[i])
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)
axes[0, 0].set_ylim([x_eq[0]-0.15,x_eq[0]+0.15])
axes[2, 0].stairs(u_nl, t_d, color='tab:red', label='Torque NL')
axes[2, 0].stairs(u_lin, t_d, color='tab:blue', label='Torque Lin')
axes[2, 0].set_ylabel('τ (N·m)')
axes[2, 0].set_title('Control torque')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].legend()
axes[2, 0].grid(True)
plt.tight_layout()
plt.show()
"""