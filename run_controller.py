from robot import RoboEnv
from mpc_controller import MPC
import numpy as np 
import matplotlib.pyplot as plt
import time


################################
# setting time is 6 seconds




def generate_save_plt(x, y1, y2, y3, y4, title, axis_names, legend, time1, time2):
    fig, axs = plt.subplots(2, 2,figsize=(10, 10))
    plt.grid(True)
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]

    # Plot each set of data in the corresponding subplot
    for i in range(y1.shape[1]):
        axs[0, 0].plot(x, y1[:, i], color=colors[i % len(colors)])
        axs[0, 1].plot(x, y2[:, i], color=colors[i % len(colors)])
        axs[1, 0].plot(x, y3[:, i], color=colors[i % len(colors)])
        axs[1, 1].plot(x, y4[:, i], color=colors[i % len(colors)])

    # Set x and y labels, titles, and limits
    axs[0, 0].set_xlabel(axis_names[0])
    axs[0, 0].set_ylabel("act force / torque")
    axs[0, 1].set_xlabel(axis_names[0])
    axs[0, 1].set_ylabel("m X ddq + c+ g")
    axs[1, 0].set_xlabel(axis_names[0])
    axs[1, 0].set_ylabel("sensor_reading")
    axs[1, 1].set_xlabel(axis_names[0])
    axs[1, 1].set_ylabel("rtb calculation")  # Change to appropriate label

    plt.legend(legend)
    axs[0, 0].set_ylim(-100, 100)  # Example y-limits
    axs[0, 1].set_ylim(-100, 100)
    axs[1, 0].set_ylim(-100, 100)
    axs[1, 1].set_ylim(-100, 100)
        # axs[1].legend(legend)

    fig.suptitle(title)

    plt.savefig('./plot_results/' + title + '.png')

def generate_save_joint_plt(x, y, title, axis_names, legend, time1, time2):
    fig, ax = plt.subplots()
    plt.grid(True)
    colors = ["green", "blue", "red", "yellow", "orange", "pink", "black"]
    for i in range(0, y.shape[1]):
        plt.plot(x, y[:, i], colors[i % len(colors)])
    ax.set_xlim(time1, time2)
    plt.title(title)
    plt.legend(legend)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/' + title + '.png')




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
    Tmax, dT = 10, 10
    mpc = MPC(dt=dT)
    curr_robot_position = robot.data.site_xpos[0]
    x, y, z, rx, ry, rz = get_path(curr_robot_position, Tmax, dT)

    motor_u, motor_output, motor_position, motor_vel, motor_time, motor_target_pose, motor_control_er = [], [], [], [], [], [], []
    start_time = time.time()
    np.random.seed(42)
    target_pose = np.random.uniform(-np.pi/2, np.pi/2, size=7)
    # target_vel = np.array([5,10.,3,-0,-0.,0,0])
    target_vel  =[0.3,0.0,0.,0,0,0.]
    ee_position = robot.get_ee_position()
    print("position",ee_position)
    target_position = [ee_position[0]+.3,ee_position[1],ee_position[2]]



    # while time.time() - start_time < Tmax:
    # curr_time = time.time() - start_time
    q, dq = robot.get_joint_positions(), robot.get_joint_vel()
    ee_pos = robot.get_ee_position()
    # mpc.run_opt_controller(target_position, target_vel,q,dq, robot)
    for i in range(0, 100):
        mpc.resolve_rate_controller(robot, target_vel)


    # wrench = np.asarray(mpc.wrench)
    time = np.asarray(mpc.time)
    act = np.asarray(mpc.act)
    T = np.asarray(mpc.T)
    W_joint = np.asarray(mpc.wrench_jac)
    joint_tor_sensor = np.asarray(mpc.joint_tor_sensor)
    rtb_torque = np.asarray(mpc.rtb_torque)


    generate_save_plt(x = time, y1 = act, y2 = T, y3 = joint_tor_sensor, y4 = rtb_torque, title =  "compare torques",axis_names= ["time", "torque"], legend =["1","2","3","4", "5", "6", "7"],time1 = time[0],time2=time[-1])


