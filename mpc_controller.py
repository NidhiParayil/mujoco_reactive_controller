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
from IPython.display import display, clear_output
import cvxpy as cp



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
        self.v, self.sensor_v, self.joint_vel_opt, self.joint_pos = [], [], [], []
        self.joint_tor_sensor, self.rtb_torque,  self.time, self.wrench, self.ee_pos_ =[],  [], [], [], []
        self.u = 0

    def get_next_torque(self, wrench, robot, target, target_vel):
        self.Y = 1
        self.Q = np.eye(self.n*2) * self.Y
        M = robot.get_M_()
    
        J = robot.get_jacobian()
        A, B = np.zeros((14,14)), np.zeros((14,7))
        A[0:7,0:7], B[0:7,0:7] = np.eye(7), np.eye(7)

        A[7:14,7:14], B[7:14,0:7]= np.eye(7), M
        dq_curr = robot.get_joint_vel()
        joint_tor = robot.get_joint_torque_sensor()
        target_vel = target_vel + np.dot(np.linalg.pinv(M),( np.dot(J.T, wrench)))
        x0 = robot.get_joint_positions()
        Q= np.eye(7)
        x_ref = np.zeros(14) 
        x_ref[0:7] = target 
        T = 2
        x = cp.Variable((14,T+1))
        u = cp.Variable((7,T)) # u is joint velocity
        cost = 0
        constr = []
        R = np.eye(7)
        P = np.eye(14)
        q = np.zeros(14)
        q[0:7] = robot.get_joint_positions()
        P = np.eye(14)*.1
        G = np.eye(7)*.2
        P = P.T @ P
        G = G.T @G
  
        for t in range(T):
            cost += cp.quad_form(x[:, t + 1] -x_ref, P)  + cp.quad_form(u[:, t], G)
            constr += [x[:, t + 1] == np.dot(A, q )+ B @ (u[:, t])]
            constr += [u[:,t] <= np.ones(7)*(1.5)]
            constr += [u[:,t] >= np.ones(7)*(-1.5)]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        print(u[:,0].value)
        print(target_vel)
        print("=====")
        return u[:,0].value 

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
        self.wrench_jac.append(np.dot(J.T, wrench))
        self.v.append(np.dot(J, robot.get_joint_vel())[0:3])
        self.sensor_v.append(robot.get_ee_vel())
        self.joint_tor_sensor.append(robot.get_joint_torque_sensor())
        self.time.append(robot.curr_time)
        self.rtb_torque.append(robot.get_rtb_joint_torque())
        self.wrench.append(wrench)  
        ee = robot.get_ee_position()
        self.ee_pos_.append([ee[0],ee[1],ee[2]])
        self.joint_vel_opt.append(self.u)
        self.joint_pos.append(robot.get_joint_positions())

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

        dt = 1
        wrench = robot.get_ee_wrench()
        target_pos = q_k  +dq_k *dt
        dq_ext = self.get_next_torque(wrench, robot, target_pos, dq_k)
        
        dq_k =  dq_ext
        self.target_pos = (q_k+ dq_k*dt)
        curr_end_eff_position = robot.get_ee_position()
        robot.run(self.target_pos)
        self.u = self.target_pos
        if curr_end_eff_position[0]>robot.ee_max_reach:
            robot.stop_robot = True

        # self.update_plot_values(robot, err)
        
    
        Torque, j_T = robot.get_joint_torque_mujoco()
        self.prev_time = robot.curr_time
        self.update_debug_values(J, wrench, Torque, robot)
        # print(self.get_next_torque(wrench, robot), self.target_pos)
    

if __name__ == '__main__':
    print("Testing MPC setup")
    mpc = MPC(0.01)
    count = 0
    while count < 10:
        mpc.run_opt_controller([1, 1, 1], [0, 0, 0], np.zeros(7), np.zeros(7), your_robot_instance)
        count += 1
    print("All good so far")