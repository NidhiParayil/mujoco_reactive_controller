from robot import RoboEnv
from mpc_controller import MPC
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_save_plt(x, y1, y2, y3, y4, title, axis_names, legend, time1, time2):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.grid(True)
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]

    # Plot each set of data in the corresponding subplot
    for i in range(y1.shape[1]):
        axs[0, 0].scatter(x, y1[:, i], color=colors[i % len(colors)], s=3)
        axs[0, 1].scatter(x, y2[:, i], color=colors[i % len(colors)], s=3)
        axs[1, 0].scatter(x, y3[:, i], color=colors[i % len(colors)], s=3)
        axs[1, 1].scatter(x, y4[:, i], color=colors[i % len(colors)], s=3)

    # Set x and y labels, titles, and limits
    axs[0, 0].set_xlabel(axis_names[0])
    axs[0, 0].set_ylabel("act force / torque")
    axs[0, 1].set_xlabel(axis_names[0])
    axs[0, 1].set_ylabel("m X ddq + c+ g")
    axs[1, 0].set_xlabel(axis_names[0])
    axs[1, 0].set_ylabel("sensor_reading")
    axs[1, 1].set_xlabel(axis_names[0])
    axs[1, 1].set_ylabel("Jac x wrench")  # Change to appropriate label

    for ax in axs.flat:
        ax.legend(legend)

    fig.suptitle(title)
    plt.savefig(f'./plot_results/{title}.png')


def generate_save_plt_ee(x, y1, y2, y3, title, axis_names, legend, time1, time2):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    plt.grid(True)
    legend_2 = ["1", "2", "3", "4", "5", "6", "7"]
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]

    # Plot each set of data in the corresponding subplot
    for i in range(y2.shape[1]):
        axs[0].plot(x, y1[:, i], color=colors[i % len(colors)])
        axs[1].plot(x, y2[:, i], color=colors[i % len(colors)])
    for i in range(y3.shape[1]):
        
        axs[2].plot(x, y3[:, i], color=colors[i % len(colors)])

    # Set x and y labels, titles, and limits
    axs[0].set_xlabel(axis_names[0])
    axs[0].set_ylabel("wrench")
    axs[1].set_xlabel(axis_names[0])
    axs[1].set_ylabel("current end eff pos")
    axs[2].set_xlabel(axis_names[0])
    axs[2].set_ylabel("control input: dq")
    axs[1].legend(legend)
    axs[2].legend(legend_2)

    fig.suptitle(title)
    plt.savefig(f'./plot_results/{title}.png')

def generate_save_plt_opt_ctrl(x, y1, y2, y3, title, axis_names, legend, time1, time2):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    plt.grid(True)
    legend_2 = ["1", "2", "3", "4", "5", "6", "7"]
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]

    # Plot each set of data in the corresponding subplot
    for i in range(y2.shape[1]):
        axs[0].plot(x, y1[:, i], color=colors[i % len(colors)])
        axs[1].plot(x, y2[:, i], color=colors[i % len(colors)])
    # for i in range(y3.shape):
    print(y3)
    axs[2].plot(x, y3, color=colors[i % len(colors)])

    # Set x and y labels, titles, and limits
    axs[0].set_xlabel(axis_names[0])
    axs[0].set_ylabel("u target")
    axs[1].set_xlabel(axis_names[0])
    axs[1].set_ylabel("u calculated")
    axs[2].set_xlabel(axis_names[0])
    axs[2].set_ylabel("cost")
    axs[1].legend(legend)
    axs[2].legend(legend_2)

    fig.suptitle(title)
    plt.savefig(f'./plot_results/{title}.png')


def get_path(robo_pos, tmax, dt):
    Nmax = int(tmax / dt)
    t = np.arange(0, Nmax) * dt
    ax, ay, az = 0.25, 0.25, 0.15
    xx = 0.1 * t + 0.1
    zz = robo_pos[2] * np.ones_like(xx)
    yy = robo_pos[1] - 0.0 * t
    rxx, rzz, ryy = -xx - 0.05, zz - 0.55, yy
    return xx, yy, zz, rxx, ryy, rzz


def mean_squared_error(x, y):
    if x.shape != y.shape:
        raise ValueError("Arrays must have the same shape")
    squared_diff = (x - y) ** 2
    return np.mean(squared_diff)


if __name__ == '__main__':
    print("testing robot.py setup")
    robot = RoboEnv()
    Tmax, dT = 10, 300
    mpc = MPC(dt=dT)
    curr_robot_position = robot.data.site_xpos[0]
    x, y, z, rx, ry, rz = get_path(curr_robot_position, Tmax, dT)

    start_time = time.time()
    np.random.seed(42)
    target_pose = np.random.uniform(-np.pi/2, np.pi/2, size=7)
    target_vel = [0.3, 0., 0., 0, 0, 0.]
    ee_position = robot.get_ee_position()
    print("position", ee_position)
    target_position = [ee_position[0] + .3, ee_position[1], ee_position[2]]
    __, _, robot_ee_ini_pose = robot.get_ee_pose()
    for i in range(0, dT):
        mpc.resolve_rate_controller(robot, target_vel, robot_ee_ini_pose)

    time_arr = np.asarray(mpc.time)
    act = np.asarray(mpc.act)
    T = np.asarray(mpc.T)
    W_joint = np.asarray(mpc.wrench_jac)
    joint_tor_sensor = np.asarray(mpc.joint_tor_sensor)
    W = np.asarray(mpc.wrench)
    ee_pos_ = np.asarray(mpc.ee_pos_)
    joint_vel_opt = np.asarray(mpc.joint_vel_opt)
    joint_pos = np.asarray(mpc.joint_pos)
    target_u = np.asarray(mpc.target_u)
    calculated_u = np.asarray(mpc.calculated_u)
    cost = np.asarray(mpc.cost)
    generate_save_plt(time_arr, act, T, joint_tor_sensor, W_joint, 
                      "compare torques with tree opt controller with force", 
                      ["time", "torque"], ["1", "2", "3", "4", "5", "6", "7"], 
                      time_arr[0], time_arr[-1])

    generate_save_plt_ee(time_arr, W, ee_pos_, joint_vel_opt, 
                         "end effector parameters with tree opt controller with force", 
                         ["time", "torque"], ["x", "y", "z"], 
                         time_arr[0], time_arr[-1])
    generate_save_plt_opt_ctrl(time_arr, target_u, calculated_u, cost, 
                         "optimization parameters with tree opt controller with force", 
                         ["time", "torque"], ["x", "y", "z"], 
                         time_arr[0], time_arr[-1])
