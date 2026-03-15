import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Model import nonLinearModel

params = {
    'Jw': 0.005, 'Jp': 0.05, 
    'mp': 0.4, 'mw': 0.2, 
    'lp': 0.3, 'lw': 0.3, 
    'b1': 0.01, 'b2': 0.005
}

plant = nonLinearModel(params)

theta = np.pi - 0.01  
theta_dot = 0.0
phi = 0.0
phi_dot = 0.0

dt = 0.05
total_time = 15.0
num_steps = int(total_time / dt)

history_theta = []
history_phi = []

print("Simulating physics...")
for _ in range(num_steps):
    history_theta.append(theta)
    history_phi.append(phi)
    
    tau = 0.0
    theta, theta_dot, phi, phi_dot = plant.next_step(theta, theta_dot, phi, phi_dot, tau, dt)

print("Building animation...")


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
    current_theta = history_theta[frame]
    current_phi = history_phi[frame]
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