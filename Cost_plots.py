import numpy as np
import matplotlib.pyplot as plt
from Model import SystemModel
from Control import Controllers
import quadprog

# System parameters
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

# Simulation setup
total_time = 10.0
num_steps = int(total_time / params['SamplingTime'])
t = np.arange(num_steps + 1) * params['SamplingTime']
yref = np.array([0, 0])

x_eq = np.array([np.pi, 0.0, 0.0, 0.0])
deviation = np.array([0.15, -0.1, 0.3, 1.0])

# State constraints
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
gx = np.hstack([0.3 * np.ones(2), 100.0 * np.ones(6)])

# Input constraints
Au = np.array([[1.0], [-1.0]])
gu = 0.5 * np.ones(2)

# MPC horizon
N = 5
R_fixed = 1.0
q_values = [1e-6, 1e-4, 1e-2, 1e-1]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Run a simulation for each value of q and store the results
all_theta = []
all_phi = []
all_torque = []

for q in q_values:
    Q = q * np.diag(np.array([100,1,0.1,0.001]))
    R = R_fixed

    controller = Controllers(A_d, B_d, C, Q, R, N, yref)
    A_con, g_con = controller.ComputeXfineq(Ax, Au, gx, gu)

    x_err = np.zeros((4, num_steps + 1))
    u = np.zeros(num_steps)
    x_err[:, 0] = deviation

    for k in range(num_steps):
        tau, _, _ = controller.mpc(x_err[:, k], Ax, gx, Au, gu, A_con, g_con)
        u[k] = tau[0]
        x_err[:, [k + 1]], _ = plant.forward_discreet_linear(x_err[:, [k]], np.array([[u[k]]]))

    x_full = x_err + x_eq[:, np.newaxis]
    all_theta.append(x_full[0, :])
    all_phi.append(x_full[2, :])
    all_torque.append(u)

    print(f"q = {q} done")

# Plot the three outputs
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Cost study', fontsize=20)

for i in range(len(q_values)):
    axes[0].stairs(all_theta[i][:-1], t, color=colors[i], linewidth=1.8, label=f'q = {q_values[i]}')
#axes[0].set_title('Pendulum angle')
axes[0].set_ylabel('theta (rad)')
#axes[0].set_xlabel('Time (s)')
axes[0].legend(fontsize=12, loc='upper right')
axes[0].grid(True, alpha=0.4)
axes[0].set_ylim([np.pi-0.3,  np.pi+0.3])
axes[0].set_xlim([0,10])

for i in range(len(q_values)):
    axes[1].stairs(all_phi[i][:-1], t, color=colors[i], linewidth=1.8, label=f'q = {q_values[i]}')
#axes[1].set_title('Wheel angle')
axes[1].set_ylabel('phi (rad)')
#axes[1].set_xlabel('Time (s)')
axes[1].legend(loc='upper right', fontsize=12)
axes[1].grid(True, alpha=0.4)
axes[1].set_xlim([0,10])
for i in range(len(q_values)):
    axes[2].stairs(all_torque[i], t, color=colors[i], linewidth=1.8, label=f'q = {q_values[i]}')
#axes[2].set_title('Control torque')
axes[2].set_ylabel('tau (N*m)')
axes[2].set_xlabel('Time (s)')
axes[2].legend(fontsize=12, loc='upper right')
axes[2].grid(True, alpha=0.4)
axes[2].set_xlim([0,10])
plt.tight_layout(pad=1.5)
plt.show()