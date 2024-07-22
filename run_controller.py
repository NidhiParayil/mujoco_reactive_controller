from robot import RoboEnv
from mpc_controller import MPC
import numpy as np 
import matplotlib.pyplot as plt
import time


################################
# setting time is 6 seconds




def generate_save_plt(dx, dy, x, y, title, axis_names, legend, time1, time2):
    fig, ax = plt.subplots()
    plt.grid(True)
    colors = ["green", "blue", "red", "yellow", "orange", "pink"]
    plt.plot(dx, dy, colors[1], label=legend[1])
    plt.plot(x, y, colors[0], label=legend[0])
    plt.title(title)
    plt.legend()
    ax.set_xlim(time1, time2)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
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
    Tmax, dT = 30, 0.2
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
    mpc.run_opt_controller(target_position, target_vel,q,dq, robot)
    # for i in range(0, 100):
    #     mpc.resolve_rate_controller(robot, target_vel)


    # wrench = np.asarray(mpc.wrench)
    # time = np.asarray(mpc.opt_ctrl.time)
    # generate_save_joint_plt(x = time, y = wrench, title = "wrench", axis_names= ["time", "force"], legend =["x","y","z","rx","ry","rz"], time1=time[0], time2=time[-1])




    
    # # Collect remaining data
    # robo_x = np.array(mpc.robo_x)
    # robo_y = np.array(mpc.robo_y)
    # robo_z = np.array(mpc.robo_z)
    # time_ = np.array(mpc.time_)
    # robo_q = np.array(mpc.robot_q)
    # robo_dq = np.array(mpc.robot_dq)
    # robo_ctl = np.array(mpc.robot_ctl)
    # err_x = np.array(mpc.robot_err_x)
    # err_y = np.array(mpc.robot_err_y)
    # err_z = np.array(mpc.robot_err_z)

    # np.savez("pid_position.npz", robo_x=robo_x, robo_y=robo_y, robo_z=robo_z, err_x=err_x, err_y=err_y, err_z=err_z, robo_q=robo_q, robo_dq=robo_dq, time_=time_, robo_ctl=robo_ctl)
