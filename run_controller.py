from robot import RoboEnv
from mpc_controller import MPC
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

def generate_save_plt(x, y_array, y_stds,  y_labels, title, legend, additional=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.grid(True)
    colors = ["green", "blue", "red", "cyan", "orange", "purple", "black"]
    colors_ad = ["black","purple","orange","cyan","green","red", "blue" ]
    plts = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]

    for y, std,  ax, lb in zip(y_array, y_stds, plts, y_labels):
        if y.ndim != 1:
            for i in range(y.shape[1]):
                # ax.scatter(x, y[:, i], color=colors[i % len(colors)], s=3)
                ax.plot(x, y[:, i], lw=2, label=legend[i], color=colors[i % len(colors)])
                ax.fill_between(x, y[:, i] + std[:,i] , y[:, i] - std[:,i], facecolor=colors[i % len(colors)], alpha=0.5)
                ax.legend()
        else:
            ax.scatter(x, y, s=3)
        ax.set_xlabel("time")
        ax.set_ylabel(lb)

    if additional is not None:
        for y, idx in zip(additional, range(len(additional))):
            if y.ndim == 2:
                for j in range(y.shape[1]):
                    plts[idx].plot(x, y[:, j], color=colors[(j % len(colors_ad))])
            else:
                plts[idx].plot(x, y)

    fig.suptitle(title)
    plt.savefig(f'./plot_results/ctrl/{title}.png')

def run_simulation(robot,controller_type, control_param):
    robot.reset_joints()
    dt, steps = 1/10, 100
    mpc = MPC(dt,controller_type, control_param)
    curr_robot_position = robot.data.site_xpos[0]
    start_time = time.time()
    np.random.seed(42)
    target_pose = np.random.uniform(-np.pi/2, np.pi/2, size=7)
    target_vel = [.1, 0., 0., 0, 0, 0.]
    ee_position = robot.get_ee_position()
    __, _, robot_ee_ini_pose = robot.get_ee_pose()
    for i in range(steps):
        mpc.resolve_rate_controller(robot, target_vel, robot_ee_ini_pose, start_time)

    results = {
        'actuator_torque': np.asarray(mpc.actuator_torque),
        'wrench_jac': np.asarray(mpc.wrench_jac),
        'time_sim': np.asarray(mpc.time),
        'rtb_torque': np.asarray(mpc.rtb_torque),
        'joint_curr_pos': np.asarray(mpc.joint_curr_pos),
        'joint_curr_vel': np.asarray(mpc.joint_curr_vel),
        'joint_curr_acc': np.asarray(mpc.joint_curr_acc),
        'joint_curr_torque': np.asarray(mpc.joint_curr_torque),
        'joint_des_pos': np.asarray(mpc.joint_des_pos),
        'joint_des_vel': np.asarray(mpc.joint_des_vel),
        'joint_des_torque': np.asarray(mpc.joint_des_torque),
        'ee_curr_pos': np.asarray(mpc.ee_curr_pos),
        'ee_curr_vel': np.asarray(mpc.ee_curr_vel),
        'ee_curr_acc': np.asarray(mpc.ee_curr_acc),
        'ee_curr_force': np.asarray(mpc.ee_curr_force),
        'ee_des_pos': np.asarray(mpc.ee_des_pos),
        'ee_des_vel': np.asarray(mpc.ee_des_vel),
        'sensor_wrench': np.asarray(mpc.sensor_wrench),
        # 'opt_u': np.asarray(mpc.opt_u),
        # 'opt_cost': np.asarray(mpc.opt_cost),
        # 'force_cost': np.asarray(mpc.force_cost),
        # 'x_cost': np.asarray(mpc.x_cost),
        # 'opt_x_ref': np.asarray(mpc.opt_x_ref),
        # 'opt_f_ref': np.asarray(mpc.opt_f_ref),
        # 'opt_error_f': np.asarray(mpc.opt_error_f),
        # 'opt_error_x': np.asarray(mpc.opt_error_x),
        # 'opt_x': np.asarray(mpc.opt_x),
        # 'opt_f': np.asarray(mpc.opt_f)
    }

    return results

def main():

    num_runs = 3
    robot = RoboEnv()
    opt_param = [100, .1, 1, .1]
    pid_pos_param = [2,0,0]
    pid_hybrid_param = [0,0,0]
    controller_types = ["pid_position", "pid_hybrid","optimizer"]
    control_parms = [pid_pos_param, pid_hybrid_param, opt_param] 
    obstacle = " plant "
    for ctrl, ctrl_param in zip( controller_types, control_parms):
        controller_type, control_param = ctrl, ctrl_param
        results_list = [run_simulation(robot,controller_type, control_param) for i in range(num_runs)]


        keys = results_list[0].keys()
        avg_results = {key: np.mean([result[key] for result in results_list], axis=0) for key in keys}
        std_devs = {key: np.std([result[key] for result in results_list], axis=0) for key in keys}


        

        time_sim = avg_results['time_sim']

        generate_save_plt(time_sim, 
                        [avg_results['ee_curr_pos'], avg_results['ee_curr_vel'], avg_results['ee_curr_force'], avg_results['ee_curr_acc']], 
                        [std_devs ['ee_curr_pos'], std_devs ['ee_curr_vel'], std_devs ['ee_curr_force'], std_devs ['ee_curr_acc']], 
                        ["pos", "vel", "force", "acc"], 
                        controller_type +obstacle+ " End effector current ", ["x", "y", "z"], 
                        [avg_results['ee_des_pos'], avg_results['ee_des_vel']])

        generate_save_plt(time_sim, 
                        [avg_results['joint_curr_pos'], avg_results['joint_curr_vel'], avg_results['joint_curr_torque'], avg_results['joint_curr_acc']], 
                        [std_devs['joint_curr_pos'], std_devs['joint_curr_vel'], std_devs['joint_curr_torque'], std_devs['joint_curr_acc']], 
                        ["pos", "vel", "force", "acc"], 
                        controller_type  +obstacle+ " Joint space " , ["1", "2", "3", "4", "5","6", "7"], None)
        
        # generate_save_plt(time_sim, 
        #                 [avg_results['opt_cost'], avg_results['opt_u'], avg_results['opt_error_f'], avg_results['opt_error_x']], 
        #                 [std_devs['opt_cost'], std_devs['opt_u'], std_devs['opt_error_f'], std_devs['opt_error_x']], 
        #                 ["total cost", "force", "x", "u"], 
        #                 controller_type + np.array2string(control_param)+obstacle + " costs", ["x", "y", "z"], None)

        # generate_save_plt(time_sim, 
        #                 [avg_results['opt_error_f'], avg_results['opt_error_x'], avg_results['opt_f'], avg_results['opt_x']], 
        #                 [std_devs['opt_error_f'], std_devs['opt_error_x'], std_devs['opt_f'], std_devs['opt_x']], 
        #                 ["force_er", "x_er", "f", "x"], 
        #                 controller_type + np.array2string(control_param)+obstacle +" f, x values and error " , ["x", "y", "z"], 
        #                 [avg_results['opt_f_ref'], avg_results['opt_x_ref']])

if __name__ == '__main__':
    main()