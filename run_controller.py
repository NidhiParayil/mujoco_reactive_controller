from robot import RoboEnv
from mpc_controller import MPC
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_save_plt(x, y_array, y_labels, time1, time2, title, legend, additional):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.grid(True)
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]
    plts = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
    # Plot each set of data in the corresponding subplot

    for y, ax, lb in zip(y_array, plts, y_labels):
        print(y)
        if y.ndim == 2:
            for i in range(y.shape[1]):
                ax.scatter(x, y[:, i], color=colors[i % len(colors)], s=3)
        else:
            ax.scatter(x, y, s=3)
        ax.set_xlabel("time")
        ax.set_ylabel(lb)
    for ax in axs.flat:
        ax.legend(legend)

    if additional is not None:
        for y, idx in zip(additional, range(len(additional))):
            if y.ndim == 2:
                for j in range(y.shape[1]):
                    plts[idx].scatter(x, y[:, j],marker="o", color=colors[j % len(colors)], s=3)
            else:
                plts[idx].scatter(x, y,marker="o", s=3)

    fig.suptitle(title)
    plt.savefig(f'./plot_results/{title}.png')





if __name__ == '__main__':
    print("testing robot.py setup")
    robot = RoboEnv()
    dt, steps = 1/10, 100
    mpc = MPC(dt=dt)
    curr_robot_position = robot.data.site_xpos[0]
    start_time = time.time()
    np.random.seed(42)
    target_pose = np.random.uniform(-np.pi/2, np.pi/2, size=7)
    target_vel = [.1, 0., 0., 0, 0, 0.]
    ee_position = robot.get_ee_position()
    # print("position", ee_position)
    __, _, robot_ee_ini_pose = robot.get_ee_pose()
    for i in range(0, steps):
        mpc.resolve_rate_controller(robot, target_vel, robot_ee_ini_pose, start_time)

    actuator_torque = np.asarray(mpc.actuator_torque)
    wrench_jac = np.asarray(mpc.wrench_jac)
    time_sim = np.asarray(mpc.time)        
    rtb_torque = np.asarray(mpc.rtb_torque)

    joint_curr_pos = np.asarray(mpc.joint_curr_pos) 
    joint_curr_vel = np.asarray(mpc.joint_curr_vel)
    joint_curr_acc = np.asarray(mpc.joint_curr_acc) 
    joint_curr_torque = np.asarray(mpc.joint_curr_torque)
    joint_des_pos = np.asarray(mpc.joint_des_pos)
    joint_des_vel = np.asarray(mpc.joint_des_vel) 
    joint_des_torque = np.asarray(mpc.joint_des_torque)

    ee_curr_pos = np.asarray(mpc.ee_curr_pos)
    ee_curr_vel = np.asarray(mpc.ee_curr_vel)
    ee_curr_acc = np.asarray(mpc.ee_curr_acc)
    ee_curr_force = np.asarray(mpc.ee_curr_force)
    ee_des_pos = np.asarray(mpc.ee_des_pos)
    ee_des_vel = np.asarray(mpc.ee_des_vel)      
    sensor_wrench = np.asarray(mpc.sensor_wrench)

    opt_u = np.asarray(mpc.opt_u)
    opt_cost = np.asarray(mpc.opt_cost)
    force_cost = np.asarray(mpc.force_cost)
    x_cost = np.asarray(mpc.x_cost)
    opt_x_ref = np.asarray(mpc.opt_x_ref)
    opt_f_ref = np.asarray(mpc.opt_f_ref)
    opt_error_f = np.asarray(mpc.opt_error_f)
    opt_error_x = np.asarray(mpc.opt_error_x)
    opt_x = np.asarray(mpc.opt_x)
    opt_f = np.asarray(mpc.opt_f)



    controller = "optimal with force"

    # To do
    # fix the desisred get_path
    # plot the errors
    # plot the optimal cpntroller

    generate_save_plt(time_sim, [ee_curr_pos, ee_curr_vel, ee_curr_force, ee_curr_acc], 
                      ["pos", "vel", "force", "acc"], 
                      time_sim[0], time_sim[-1], "End effector current"+controller,["x","y","z"], [ee_des_pos, ee_des_vel])

    generate_save_plt(time_sim, [joint_curr_pos,joint_curr_vel,joint_curr_torque,joint_curr_acc], 
                      ["pos", "vel", "force", "acc"], 
                      time_sim[0], time_sim[-1], " joint space"+ controller,["1","2","3","4","5","7"], None)
    generate_save_plt(time_sim, [opt_cost,force_cost,x_cost,opt_u], 
                      ["total cost", "force", "x", "u"], 
                      time_sim[0], time_sim[-1], "cost"+ controller,["- "], None)


    generate_save_plt(time_sim, [opt_error_f, opt_error_x, opt_f, opt_x], 
                    ["force_er", "x_er", "f", "x"], 
                    time_sim[0], time_sim[-1], "f, x values and error"+ controller,["- "], [opt_f_ref, opt_x_ref])