import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Model import SystemModel 
from Control import Controllers
import control as ct

params = {
    'Jw': 0.005, 'Jp': 0.05, 
    'mp': 0.4, 'mw': 0.2, 
    'lp': 0.3, 'lw': 0.3, 
    'b1': 0.01, 'b2': 0.005,
    'SamplingTime': 0.1, 'Theta_eq': np.pi
}


plant = SystemModel(params)

A, B, C = plant.Linearised(params['Theta_eq'])
A_d, B_d = plant.ZeroOrderHold(params['SamplingTime'])
Q_val = np.array([50, 0.01, 0.01, 0.01])
Q = np.diag(Q_val)
R = 100
N = 5
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
    
    
x_nl[:, 0] = x0
x_lin_err[:, 0] = deviation
x_obs[:,0] = deviation
x_obs_nl[:,0] = deviation

theta, theta_dot, phi, phi_dot = x0[0], x0[1], x0[2], x0[3]
#print(x_target, u_target)
y_nl = np.array([[deviation[0]],[deviation[2]]])
for i in range(num_steps):
    if i % calc_u_count == 0:
        k = i // calc_u_count
        if k == num_steps_dis//2:
            d = np.array([[0.01]])
        error_nl = x_obs_nl[:, k]
        tau_nl = Controller.mpc(error_nl,Ax, gx, Au, gu, A_con, g_con, dist=d_nl[:,[k]])[0]  
        tau_lin = Controller.mpc(x_obs[:,k],Ax, gx, Au, gu, A_con, g_con, dist=d_lin[:,[k]])[0]
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


t = np.arange(num_steps + 1) * dt
t_d = np.arange(num_steps_dis + 1) * params['SamplingTime']
u_lin = np.array(u_lin)

fig, axes = plt.subplots(4, 2, figsize=(12, 14))
fig.suptitle('Observer-based MPC Response')

labels = ['θ error (rad)', 'θ̇ error (rad/s)', 'φ error (rad)', 'φ̇ error (rad/s)']
titles = ['Pendulum angle error', 'Pendulum angular velocity error', 'Wheel angle error', 'Wheel angular velocity error']

for i in range(4):
    axes[i, 0].stairs(x_lin_err[i, :-1], t_d, label='True error')
    axes[i, 0].stairs(x_obs_nl[i, :-1], t_d, label='Observed NL')
    axes[i, 0].stairs(x_obs[i, :-1], t_d, linestyle='--', label='Observed')
    axes[i, 0].plot(t, x_nl[i], label='NL Model')
    axes[i, 0].set_ylabel(labels[i])
    axes[i, 0].set_title(titles[i])
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].legend()
    axes[i, 0].grid(True)

axes[0, 1].stairs(d_lin[0, :-1], t_d, color='tab:orange', label='Estimated disturbance')
axes[0, 1].stairs(d_nl[0, :-1], t_d, color='tab:orange', label='Estimated disturbance NL')
axes[0, 1].axhline(d[0, 0], color='k', linestyle='--', label=f'True disturbance ({d[0,0]})')
axes[0, 1].set_ylabel('d')
axes[0, 1].set_title('Disturbance estimate')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 1].stairs(u_lin, t_d, color='tab:red', label='Control torque')
axes[1, 1].stairs(u_nl, t_d, color='tab:blue', label='Control torque nl')
axes[1, 1].set_ylabel('τ (N·m)')
axes[1, 1].set_title('Control torque')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[2, 1].stairs(y_lin_err[0, :], t_d, label='y1 (theta)')
axes[2, 1].stairs(y_obs[0, :], t_d, linestyle='--', label='y1 observed')
axes[2, 1].set_ylabel('y1')
axes[2, 1].set_title('Output y1 (θ)')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].legend()
axes[2, 1].grid(True)

axes[3, 1].stairs(y_lin_err[1, :], t_d, label='y2 (phi)')
axes[3, 1].stairs(y_obs[1, :], t_d, linestyle='--', label='y2 observed')
axes[3, 1].set_ylabel('y2')
axes[3, 1].set_title('Output y2 (φ)')
axes[3, 1].set_xlabel('Time (s)')
axes[3, 1].legend()
axes[3, 1].grid(True)

plt.tight_layout()
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