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
    def __init__(self, dt):
        self.n = 7  # Number of joints
        self.dt = dt
        self.initialize_storage()
        self.opt_ctrl = ControlPerformance("opt_ctrl")
        time.sleep(1)
        print("Start controller")
        self.Ax, self.Bx, self.Af, self.Bf = np.eye(3), np.eye(3), np.eye(3), np.eye(3)
        self.R = np.eye(self.n)
        self.P = np.eye(3)*2
        self.F = np.eye(3)*100
        self.G = np.eye(3)*1
        self.P = self.P.T @ self.P
        self.G = self.G.T @self.G
    

    def initialize_storage(self):
        self.joint_curr_pos, self.joint_curr_vel, self.joint_curr_acc, self.joint_curr_torque = [], [], [], []
        self.joint_des_pos, self.joint_des_vel, self.joint_des_acc, self.joint_des_torque = [], [], [], []
        self.ee_curr_pos, self.ee_curr_vel, self.ee_curr_acc, self.ee_curr_force = [], [], [], []
        self.ee_des_pos, self.ee_des_vel, self.ee_des_acc, self.ee_des_force = [], [], [], []
        self.opt_error_x, self.opt_ref_x, self.opt_x, = [], [], []
        self.opt_error_f, self.opt_ref_f, self.opt_f, = [], [], []
        self.opt_u , self.opt_cost = [], []
        self.sensor_acc, self.sensor_vel, self.sensor_wrench, self.sensor_jointTor = [], [], [], []
        


    def get_optimal_vel(self, wrench, robot, target_pos, target_vel):

        f_ref = wrench
        x_ref = target_pos 
        self.T_opt = 2
        x = cp.Variable((3,self.T_opt+1))
        f = cp.Variable((3,self.T_opt+1))
        u = cp.Variable((3,self.T_opt)) # u is joint velocity
        cost = 0
        constr = []   
        x0 = robot.get_ee_position()
        for t in range(self.T_opt):
            cost += cp.quad_form(x[:, t + 1] -x_ref, self.P) + cp.quad_form(f[:, t + 1] -f_ref, self.F) + cp.quad_form(u[:, t] , self.G)
            constr += [x[:, t + 1] == np.dot(self.Ax, x0 )+ self.Bx @ (u[:, t])]
            constr += [f[:, t + 1] == self.Bf @ (u[:, t])]
            constr += [u[:,t] <= np.ones(3)*(1.5)]
            constr += [u[:,t] >= np.ones(3)*(-1.5)]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        self.opt_u.append(u[:,0].value)
        self.opt_cost.append(problem.value)
        force_cost = np.matmul((f[:, 0].value -f_ref).T,np.matmul(self.F, f[:, 0].value -f_ref) )
        pos_cost = np.matmul((x[:, 0].value -x_ref).T,np.matmul(self.P, x[:, 0].value -x_ref) )
        return u[:,0].value 


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

    def resolve_rate_controller(self, robot, desired_ee_vel, robot_ee_pose):
        J = robot.get_jacobian()
        pnv_j = np.linalg.pinv(J)
        next_ee_vel = np.zeros(6)
        desired_ee_vel = np.asarray(desired_ee_vel)[0:3]
        desired_ee_position = robot_ee_pose[3, 0:3] + desired_ee_vel *robot.curr_time
        wrench = robot.get_ee_wrench()
        next_ee_vel[0:3] = self.get_optimal_vel(wrench[0:3], robot, desired_ee_position, desired_ee_vel)
        q_curr = robot.get_joint_positions()
        dq_next = np.matmul(pnv_j, next_ee_vel)
        self.target_pos = (q_curr+ dq_next)
        curr_end_eff_position = robot.get_ee_position()
        robot.run(self.target_pos)
        self.u = self.target_pos
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