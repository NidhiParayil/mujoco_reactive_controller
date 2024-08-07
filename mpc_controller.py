import time
import numpy as np
from urdfpy import URDF
from spatialmath import SE3
import spatialmath as sm
import roboticstoolbox as rtb
import spatialmath.base as base
from controller_performance import ControlPerformance
import cvxpy as cp



class MPC:
    def __init__(self, dt, controller_type, control_param):
        self.n = 7  # Number of joints
        self.dt_opt = dt
        self.dt= dt
        self.opt_ctrl = ControlPerformance("opt_ctrl")
        time.sleep(1)
        print("Start controller")
        self.controller_type = controller_type
        self.initialize_storage()
        self.set_contntroll_param(control_param)




    def set_contntroll_param(self, param):

        if self.controller_type == "optimizer":
            self.Ax, self.Bx, self.Af, self.Bf = np.eye(3), np.eye(3)*self.dt_opt, np.eye(3), np.eye(3)/self.dt_opt
            self.P = np.eye(3)*param[0]
            self.F = np.eye(3)*param[1]
            self.G = np.eye(3)*param[2]
            self.H = np.eye(3)*param[3]

            self.P = self.P.T @ self.P
            self.G = self.G.T @self.G
            self.F = self.F.T @self.F
            self.H = self.H.T @self.H

            self.prev_u = np.zeros(3)

        if self.controller_type == "pid_position":
            self.kp_pos = param[0]
            self.kd_pos = param[1]
            self.ki_pos = param[2]

        if self.controller_type == "pid_hybrid":
            pass

    

    def initialize_storage(self):
        self.joint_curr_pos, self.joint_curr_vel, self.joint_curr_acc, self.joint_curr_torque = [], [], [], []
        self.joint_des_pos, self.joint_des_vel, self.joint_des_acc, self.joint_des_torque = [], [], [], []
        self.ee_curr_pos, self.ee_curr_vel, self.ee_curr_acc, self.ee_curr_force = [], [], [], []
        self.ee_des_pos, self.ee_des_vel, self.ee_des_acc, self.ee_des_force = [], [], [], []
        self.actuator_torque, self.rtb_torque, self.sensor_wrench, self.time, self.wrench_jac = [], [], [], [], []

        if self.controller_type == "optimizer":
            self.opt_error_x, self.opt_x_ref, self.opt_x, = [], [], []
            self.opt_error_f, self.opt_f_ref, self.opt_f, = [], [], []
            self.opt_u , self.opt_cost, self.force_cost, self.x_cost = [], [], [], []
        if self.controller_type == "pid_position":
            self.pid_error_x, self.pid_x_ref, self.pid_u = [], [], []
            self.error_sum = 0
            self.error_prev = 0


    def get_pid_pos_values(self, error, target):
        self.error_sum = error + self.error_sum
        u =  self.kp_pos*error + self.kd_pos/self.dt * (error-self.error_prev) + self.ki_pos * self.error_sum
        self.error_prev = error
        self.pid_error_x.append(error)
        self.pid_x_ref.append(target)
        self.pid_u.append(u)
        return u 

    def get_optimal_vel(self, wrench, robot, target_pos, target_vel):
        f_ref = wrench
        x_ref = np.asarray(target_pos) 
        self.T_opt = 2
        x = cp.Variable((3,self.T_opt+1))
        f = cp.Variable((3,self.T_opt+1))
        u = cp.Variable((3,self.T_opt)) # u is joint velocity
        cost = 0
        constr = []   
        x0 = robot.get_ee_position()
        v0 = robot.get_ee_vel()
        # print("wrench", wrench)
        for t in range(self.T_opt):
            cost += cp.quad_form(x[:, t + 1] -x_ref, self.P) + cp.quad_form(f[:, t + 1] -f_ref, self.F) + cp.quad_form(u[:, t] , self.G) + cp.quad_form(u[:, t] -self.prev_u, self.H)
            constr += [x[:, t + 1] == self.Ax @ x0 + self.Bx @ (u[:, t])]
            constr += [f[:, t + 1] == self.Bf @ (u[:, t]- v0)]
            constr += [u[:,t] <= np.ones(3)*(3)]
            constr += [u[:,t] >= np.ones(3)*(-3)]
        problem = cp.Problem(cp.Minimize(cost), constr)
        # print(x0)
        # print(target_pos)
        problem.solve()
        x_k_1  = np.dot(self.Ax, x0 )+np.dot( self.Bx,(u[:, 0].value))
        f_k_1 = np.dot(self.Bf, (u[:, 0].value- v0))
        x_er = x_k_1 - x_ref
        f_er = f_k_1 - f_ref
        self.opt_u.append(u[:,0].value)
        self.opt_cost.append(problem.value)
        self.force_cost.append(np.matmul((f_er).T,np.matmul(self.F, f_er)))
        self.x_cost.append(np.matmul((x_er).T,np.matmul(self.P, x_er)))
        self.opt_x_ref.append(x_ref)
        self.opt_f_ref.append(f_ref)
        self.opt_error_f.append(f_er)
        self.opt_error_x.append(x_er)
        self.opt_x.append(x_k_1)
        self.opt_f.append(f_k_1)
        self.prev_u = u[:,0].value 
        return u[:,0].value 


    def update_debug_values(self, J, Torque, robot, target_dq, target_q, desired_ee_position, desired_ee_vel):
        self.actuator_torque.append(robot.get_joint_torque_mujoco()[1])
        self.wrench_jac.append(np.dot(J.T, robot.get_ee_wrench()))
        self.time.append(robot.curr_time)
        self.rtb_torque.append(robot.get_rtb_joint_torque())
        self.joint_curr_pos.append(robot.get_joint_positions()) 
        self.joint_curr_vel.append(robot.get_joint_vel()) 
        self.joint_curr_acc.append(robot.get_joint_acc()) 
        self.joint_curr_torque.append(robot.get_joint_torque_sensor())
        self.joint_des_pos.append(target_q)
        self.joint_des_vel.append(target_dq)
        # self.joint_des_acc.append(target_ddq)
        self.joint_des_torque.append(Torque)
        ee_curr_pos = robot.get_ee_position()
        self.ee_curr_pos.append([ee_curr_pos[0],ee_curr_pos[1],ee_curr_pos[2]])
        # print(robot.get_ee_position())
        self.ee_curr_vel.append(robot.get_ee_vel())
        self.ee_curr_acc.append(robot.get_ee_acc())
        self.ee_curr_force.append(robot.get_ee_acc())
        self.ee_des_pos.append(desired_ee_position)
        self.ee_des_vel.append(desired_ee_vel)
        self.sensor_wrench.append(robot.get_ee_wrench())
               

    def resolve_rate_controller(self, robot, desired_ee_vel, robot_ee_pose, start_time):
        J = robot.get_jacobian()
        pnv_j = np.linalg.pinv(J)
        next_ee_vel = np.zeros(6)
        desired_ee_vel = np.asarray(desired_ee_vel)[0:3]
        curr_end_eff_position = robot.get_ee_position()
        # print(robot_ee_pose)
        robot_ee_pose = [.12,0,.3]
        # print(curr_end_eff_position)
        
        wrench = robot.get_ee_wrench()
        force = np.asarray(wrench[0:3])
        if self.controller_type == "optimizer":
            desired_ee_position = curr_end_eff_position + desired_ee_vel * self.dt_opt#*(time.time()-start_time)
            next_ee_vel[0:3] = self.get_optimal_vel(force, robot, desired_ee_position, desired_ee_vel)
            q_curr = robot.get_joint_positions()
            target_dq = np.matmul(pnv_j, next_ee_vel)
            target_q = (q_curr+ target_dq)

        if self.controller_type == "pid_position":
            desired_ee_position = robot_ee_pose + desired_ee_vel *(time.time()-start_time)
            ee_pos = np.zeros(6)
            error = (desired_ee_position - curr_end_eff_position)
            # print(error)
            ee_pos[0:3] = self.get_pid_pos_values(desired_ee_position,error )
            delta_q = np.matmul(J.T,ee_pos)
            target_q  = delta_q + robot.get_joint_positions()
            target_dq= ee_pos[0:3]/self.dt


        start_time = time.time()
        while (time.time() - start_time < self.dt_opt):
            robot.run(target_q)
        Torque = robot.get_joint_torque_mujoco()
        self.update_debug_values(J, Torque, robot, target_dq, target_q, desired_ee_position, desired_ee_vel)
        if curr_end_eff_position[0]>robot.ee_max_reach:
            robot.stop_robot = True

if __name__ == '__main__':
    print("Testing MPC setup")
    mpc = MPC(0.01)
    count = 0
    while count < 10:
        mpc.run_opt_controller([1, 1, 1], [0, 0, 0], np.zeros(7), np.zeros(7), your_robot_instance)
        count += 1
    print("All good so far")