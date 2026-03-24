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
Q_val = np.array([5000, 0.01, 0.01, 0.01])
Q = np.diag(Q_val)
R = 1
N = 5

Controller = Controllers(A_d, B_d, C, Q, R, N)

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
x = np.zeros((4,num_steps+1))
y = np.zeros((2,num_steps))
u = []

x_nl = np.zeros((4, num_steps + 1))
deviation = np.array([-0.3, 0.5, -0.2, 0.])
if params['Theta_eq'] == 0:
    x_eq = np.array([0., 0., 0., 0.])
    x0 = x_eq+deviation
else:
    x_eq = np.array([np.pi, 0., 0., 0.])
    x0 = x_eq+deviation
    
    
x_nl[:, 0] = x0

theta, theta_dot, phi, phi_dot = x0[0], x0[1], x0[2], x0[3]

calc_u_count = int(params["SamplingTime"] / dt)


for i in range(num_steps):
    if i % calc_u_count == 0:
        error = x_nl[:, i] - x_eq
        tau = Controllers.mpc(error)[0]
        u.append(tau)
        #x[:,i+1], y[:,i] = plant.forward_discreet_linear(x[:,i], tau)
    
    # Non linear Model
    theta, theta_dot, phi, phi_dot = plant.next_step_nonlinear(theta, theta_dot, phi, phi_dot, tau, dt)
    x_nl[:, i+1] = [theta, theta_dot, phi, phi_dot]
u = np.array(u)
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
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
fig.suptitle('Linear vs Nonlinear - MPC Response')

for i, ax in enumerate(axes.flat[:4]):
    ax.plot(t, x_nl[i], linestyle='--', label='Nonlinear')
    ax.set_ylabel(labels[i])
    ax.set_title(titles[i])
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

axes[2, 0].stairs(u, t_d, color='tab:red', label='Torque')
axes[2, 0].set_ylabel('τ (N·m)')
axes[2, 0].set_title('Control torque')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].set_visible(False)
plt.tight_layout()
plt.show()