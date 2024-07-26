import time
import cvxopt
import numpy as np
from urdfpy import URDF
from spatialmath import SE3
import spatialmath as sm
import roboticstoolbox as rtb
import spatialmath.base as base
from qpsolvers import solve_qp
from uncMPC import uncMPC
import qpsolvers as qp
from controller_performance import ControlPerformance
import mpc_qp


class MPC:
    def __init__(self, dt):
        cvxopt.solvers.options['show_progress'] = False
        self.n = 7  # Number of joints
        self.dt = dt
        self.stop_time = dt
        self.q0 = np.zeros(self.n)
        self.initialize_constraints()
        self.initialize_storage()
        self.opt_ctrl = ControlPerformance("opt_ctrl")
        time.sleep(1)
        print("Start controller")

    def initialize_constraints(self):
        self.u_UB = np.array([10, 10, 10, 10, 10, 10, 10])
        self.u_LB = -self.u_UB
        self.q_UB = np.array([6.28319, 6.28319, 2.61799, 6.28319, 6.28319, 6.28319, 6.28319])
        self.q_LB = -self.q_UB
        self.joint_configs = np.zeros(self.n)
        self.prev_u = np.zeros(8)

    def initialize_storage(self):
        self.arrived = False
        self.robo_x, self.robo_y, self.robo_z = [], [], []
        self.robo_fx, self.robo_fy, self.robo_fz = [], [], []
        self.time_, self.robot_q, self.robot_dq = [], [], []
        self.robot_ctl, self.mpc_ctl, self.mpc_cost_Q = [], [], []
        self.mpc_cost_c, self.mpc_cost_Aeq, self.mpc_cost_Beq = [], [], []
        self.mpc_cost_Ain, self.mpc_cost_Bin, self.MPC_obj = [], [], []
        self.wrench_sample = np.zeros((5, 6))
        self.A = np.zeros((self.n * 2, self.n * 2))
        self.robot_err_x, self.robot_err_y, self.robot_err_z = [], [], []
        self.act, self.T, self.wrench_jac = [], [], []
        self.v, self.sensor_v = [], []
        self.joint_tor_sensor, self.rtb_torque,  self.time = [], [], []

    def run_opt_controller(self, target_position, target_vel, q, dq, robot):
        val, pos, oreint = robot.get_rtb_end_eff_pose()
        i = 0
        self.arrived = False
        count_run_mpc = 0
        wrench = [0.01] * 7
        self.prev_time = robot.curr_time
        robot.curr_time = time.time() - robot.start_time
        self.wrench = []

        while not self.arrived:
            count_run_mpc += 1
            q, dq = robot.get_joint_positions(), robot.get_joint_vel()
            robot.curr_time = time.time() - robot.start_time
            dt = robot.curr_time - self.prev_time

            wrench = robot.get_ee_wrench()

            M = robot.get_M_()
            M_inv = np.linalg.inv(M)
            self.wrench.append(wrench)
            self.update_wrench_sample(wrench)
            wrench_avg = np.mean(self.wrench_sample, axis=0)

            J = robot.get_jacobian()
            if robot.curr_time > self.stop_time:
                self.arrived = True
                print("Controller stopping")

            self.Y = 0.1
            self.Q = np.eye(self.n) * self.Y

            acc = self.compute_acceleration(wrench, robot, J, dq)

            Aeq, beq = self.setup_constraints(J, target_vel, robot)
            Torque, j_T = robot.get_joint_torque_mujoco()
            self.update_debug_values(J, wrench, Torque, robot)

            Ain, bin = self.joint_velocity_damper(robot)

            lb, ub = self.q_LB, self.q_UB

            c = np.zeros(self.n)
            dq_d = qp.solve_qp(self.Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='cvxopt')

            self.execute_control(dq_d, robot, target_position, target_vel, q, dq)

            self.prev_time = robot.curr_time

        self.finalize_control(dq_d, robot)
        self.opt_ctrl.convert_np()
        self.opt_ctrl.plot_performance()

    def update_wrench_sample(self, wrench):
        self.wrench_sample = np.roll(self.wrench_sample, shift=-1, axis=0)
        self.wrench_sample[-1] = wrench

    def compute_acceleration(self, wrench, robot, J, dq):
        if np.abs(np.mean(wrench)) > 1:
            ddq = robot.get_forward_dynamics(wrench)
            acc = np.matmul(J, ddq)
        else:
            acc = np.zeros(6)
        return acc

    def setup_constraints(self, J, target_vel, robot):
        Aeq = np.zeros((6, self.n))
        beq = np.zeros(6)
        Aeq[0:6, 0:7] = J
        beq[0:6] = target_vel
        return Aeq, beq

    def update_debug_values(self, J, wrench, Torque, robot):
        self.T.append(Torque)
        self.act.append(robot.get_joint_torque_mujoco()[1])
        self.wrench_jac.append(np.matmul(J.T, wrench))
        self.v.append(np.dot(J, robot.get_joint_vel())[0:3])
        self.sensor_v.append(robot.get_ee_vel())
        self.joint_tor_sensor.append(robot.get_joint_torque_sensor())
        self.time.append(robot.curr_time)
        self.rtb_torque.append(robot.get_rtb_joint_torque())

    def joint_velocity_damper(self, robot):
        ps, pi = 0.05, 0.9
        return robot.robot.joint_velocity_damper(ps, pi, self.n)

    def execute_control(self, dq_d, robot, target_position, target_vel, q, dq):
        if dq_d is None:
            u = self.prev_u[0:7]
            robot.run(np.zeros(7))
            self.mpc_ctl.append(self.prev_u)
        else:
            u = dq_d[0:7] * 60
            robot.run(u)
            self.prev_u = u
            self.mpc_ctl.append(u[:7])

        ee_pos = robot.get_ee_position()
        self.opt_ctrl.append_values(
            u=robot.data.ctrl,
            y=[ee_pos[0], ee_pos[1], ee_pos[2]],
            x=q,
            dx=dq,
            target_x=target_position,
            target_dx=target_vel,
            T=robot.data.actuator_force,
            e=np.subtract(target_position, ee_pos),
            t=robot.curr_time
        )

    def finalize_control(self, dq_d, robot):
        if dq_d is None:
            robot.run(self.prev_u[0:7])
        else:
            u = dq_d[0:7] * self.dt * 60
            self.prev_u = u[0:7]
            robot.run(u[0:7])

    def resolve_rate_controller(self, robot, target):
        J = robot.get_jacobian()
        pnv_j = np.linalg.pinv(J)

        target_vel = target
        q_k = robot.get_joint_positions()
        dq_k = np.matmul(pnv_j, target_vel)
        err = [0, 0, 0]
        dt = 1.0 / 60.0
        self.target_pos = q_k +  dq_k *60
        robot.run(self.target_pos)

        # self.update_plot_values(robot, err)
        wrench = robot.get_ee_wrench()
        Torque, j_T = robot.get_joint_torque_mujoco()
        self.prev_time = robot.curr_time
        self.update_debug_values(J, wrench, Torque, robot)
        

    def update_plot_values(self, robot, err):
        self.robo_x.append(robot.data.site_xpos[0][0])
        self.robo_y.append(robot.data.site_xpos[0][1])
        self.robo_z.append(robot.data.site_xpos[0][2])
        self.time_.append(time.time())
        self.robot_q.append(robot.get_joint_positions())
        self.robot_dq.append(robot.get_joint_vel())
        self.robot_ctl.append(robot.data.ctrl)
        self.robot_err_x.append(err[0])
        self.robot_err_y.append(err[1])
        self.robot_err_z.append(err[2])


if __name__ == '__main__':
    print("Testing MPC setup")
    mpc = MPC(0.01)
    count = 0
    while count < 10:
        mpc.run_opt_controller([1, 1, 1], [0, 0, 0], np.zeros(7), np.zeros(7), your_robot_instance)
        count += 1
    print("All good so far")