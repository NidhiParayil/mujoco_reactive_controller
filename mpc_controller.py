
import time
import cvxopt
import numpy as np
from urdfpy import URDF
from spatialmath import SE3
import spatialmath as sm
import roboticstoolbox as rtb
import spatialmath.base as base
from qpsolvers import solve_problem
from uncMPC import uncMPC
import qpsolvers as qp

from controller_performance import ControlPerformance

import mpc_qp


class MPC():

    def __init__(self, dt):
        cvxopt.solvers.options['show_progress'] = False
        # MPC Parameters
        self.n = 7  # Number of joints 
        self.dt = dt
        # self.tmax = tmax
        # MPC setup
        self.q0 = np.zeros(self.n)

        # Robot-specific Constraints
        self.u_UB = np.array([10, 10, 10,10, 10, 10, 10]) # Max speed (degree/s)
        self.u_LB = -self.u_UB
        self.q_UB = np.array([6.28319, 6.28319, 2.61799, 6.28319, 6.28319, 6.28319,6.28319]) # Motion range (rad)
        self.q_LB = -self.q_UB
        self.joint_configs = np.zeros(self.n)
        self.arrived = False
        self.prev_u = np.zeros(8)
        self.robo_x = []
        self.robo_y = []
        self.robo_z = []
        self.robo_fx = []
        self.robo_fy = []
        self.robo_fz = []
        self.time_ = []
        self.robot_q = []
        self.robot_dq = []
        self.robot_ctl = []
        self.mpc_ctl = []
        self.mpc_cost_Q = []
        self.mpc_cost_c = []
        self.mpc_cost_Aeq =[]
        self.mpc_cost_Beq =[]
        self.mpc_cost_Ain =[]
        self.mpc_cost_Bin =[]
        self.MPC_obj = []
        self.wrench_sample = np.zeros((5, 6))
        self.A = np.zeros((self.n*2,self.n*2))
        
        self.opt_ctrl = ControlPerformance("opt_ctrl")
        self.robot_err_x = []
        self.robot_err_y = []
        self.robot_err_z = []
        # self.
        time.sleep(1)
        print("start controller")



    def run_opt_controller(self,target_position, target_vel, q,dq, robot):
        val , pos, oreint =   robot.get_rbt_end_eff_pose()
        i = 0
        # self.pre_error =10
        self.arrived = False
        count_run_mpc = 0
        wrench = [0.01,0.01,0.01,0.01,0.01,0.01,.01]
        self.prev_time = robot.curr_time
        robot.curr_time = time.time()- robot.start_time
        self.wrench =[]

        while not self.arrived:
            count_run_mpc  = count_run_mpc +1
            q, dq = robot.get_joint_positions(), robot.get_joint_vel()
            
            robot.curr_time = time.time()- robot.start_time
            dt = robot.curr_time - self.prev_time
            # print("wrench", np.abs(np.mean(wrench)))

            wrench = robot.get_ee_wrench()
            M = robot.get_M_()
            M_inv = np.linalg.inv(M)
            self.wrench.append(wrench)
            self.wrench_sample = np.roll(self.wrench_sample, shift=-1, axis=0)
            self.wrench_sample[-1] = wrench
            wrench_avg = np.array([sum(col) / len(col) for col in zip(*self.wrench_sample)])

            J  = robot.get_jacobian()

            
            if robot.curr_time>50:
                self.arrived = True
                print("should not enter")

            self.Y = 0.1

            self.Q = np.eye(self.n)
            self.Q[:self.n, :self.n] *= self.Y

            if np.abs(np.mean(wrench))>1:
                # print("huge wrench")
                ddq = robot.get_forward_dynamics(wrench)
                # J_dot = 
                acc = np.matmul(J, ddq) #+ np.matmul(J_dot, dq)

            else:
                acc = np.zeros(6)
                # print("no wrench")

            # dq_k_1 = np.asarray(dq) + np.asarray(ddq)*(1/60) 
            # pnv_j = np.linalg.pinv(J)
            # vel_k_1 = np.dot(J, dq_k_1)
            Aeq = np.zeros((6,self.n))
            beq = np.zeros(6)  
            # print(Aeq)        
            Aeq[0:6,0:7] = robot.get_jacobian()
            # Aeq[6:12,7:14] = robot.get_jacobian()
            # print(vel_k_1)
            beq[0:6]= np.asarray(target_vel) 
            print("--------------------")
            Torque = robot.get_joint_torque()
            
            force_ex = np.round(wrench - np.dot(J, Torque), 1)
            # beq[6:]=force_ex
            J_inv = np.linalg.pinv(J)
            # print("wrench", np.round( np.dot(J.T, wrench),2))
            print("wrench diff ", np.round((np.dot(J.T, wrench) - Torque),2))
            print("test jaco", np.dot(J, dq)- robot.data.efc_vel[-1])

            Ain = np.zeros((self.n, self.n))
            bin = np.zeros(self.n)

            # The minimum angle (in radians) in which the joint is allowed to approach
            # to its limit
            ps = 0.05

            # The influence angle (in radians) in which the velocity damper
            # becomes active
            pi = 0.9

            # Form the joint limit velocity damper
            Ain, bin= robot.robot.joint_velocity_damper(ps, pi, self.n)
                # Linear component of objective function: the manipulability Jacobian
            # c = np.r_[-np.delete(np.array(robot.robot.jacobe(robot.robot.q)), -1, axis = 1).reshape((n,)), np.zeros(6)]

            # The lower and upper bounds on the joint velocity 
            lb = self.q_LB
            ub = self.q_UB
            
            
            # c = np.matmul(np.transpose(J), wrench)*1000
    
            c = np.zeros(self.n)
            dq_d = qp.solve_qp(self.Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver = 'cvxopt')  
  
            q= robot.get_joint_positions()  
            ee_pos = robot.get_ee_position()   
            
            if dq_d is None:
                u = self.prev_u[0:7]
                robot.run(np.zeros(7))
                self.mpc_ctl.append(self.prev_u)
            else:
                # qd[5] = 0
                # qd[6] = 0
                # qd[7] = 0
                u = dq_d[0:7]*60 #+ np.dot(M_inv,dq_d[7:14])*1200*dt*dt/2
                robot.run(u)
                self.prev_u =u
                self.mpc_ctl.append(u[:7])




            self.target_pos = u
            self.opt_ctrl.append_values(
                u =robot.data.ctrl, 
                y = [ee_pos[0],ee_pos[1],ee_pos[2]], 
                x = q, 
                dx = dq, 
                target_x =target_position, 
                target_dx = target_vel, 
                T = robot.data.actuator_force, 
                e = np.subtract(target_position, ee_pos), 
                t = robot.curr_time
            )
            self.prev_time = robot.curr_time

            # if qd is None:
            #     qd = [0,0,0,0,0,0,0]

            # term1 = np.dot(np.transpose(qd), np.dot(self.Q,qd))
            # term2 = np.dot(np.transpose(c), qd)
            # mpc_obj = term1 +term2

            # self.update_plot_values(robot, mpc_obj, wrench, Ain, bin, Aeq, beq, term1, term2, c)
            # if robot.data.site_xpos[0][0]>.6:
            #     break

            
        if dq_d is None:
            robot.run(self.prev_u[0:7])
    
        else:
            u = dq_d[0:7]*dt*60 #+np.dot(M_inv,dq_d[7:14])*1200/2
            self.prev_u =u[0:7]
            robot.run(u[0:7])
        # return qd[:n], qd[:n], self.Tg,self.arrived
        self.opt_ctrl.convert_np()
        self.opt_ctrl.plot_performance()
        


    # Move at a constant end effectotr speed 

    def resolve_rate_controller(self,robot,target):
        
        J  = robot.get_jacobian()
        pnv_j = np.linalg.pinv(J)

        target_vel = target
        q_k = robot.get_joint_positions()
        dq_k = np.matmul(pnv_j,target_vel)
        err = [0,0,0]
        dt =1.0/60.0
        self.target_pos = q_k + dt*dq_k
        robot.run(self.target_pos)


        self.update_plot_values(robot, err)
 
        


    # def update_plot_values(self, robot, mpc_obj, wrench, Ain, bin, Aeq, beq, term1, term2, c):
    def update_plot_values(self, robot, err):
        # self.MPC_obj.append([mpc_obj, term1, term2])
        

        self.robo_x.append(robot.data.site_xpos[0][0])
        self.robo_y.append(robot.data.site_xpos[0][1])
        self.robo_z.append(robot.data.site_xpos[0][2])
        # self.robo_fx.append(np.linalg.norm(wrench[0:3]))
        # self.robo_fy.append(wrench[1])
        # self.robo_fz.append(wrench[2]) 
        self.time_.append(time.time())
        self.robot_q.append(robot.get_joint_positions())
        self.robot_dq.append(robot.get_joint_vel())
        self.robot_ctl.append(robot.data.ctrl)
        
        self.robot_err_x.append((err[0]))
        self.robot_err_y.append((err[1]))
        self.robot_err_z.append((err[2]))
 
        # self.mpc_cost_Q.append(self.Q)
        # self.mpc_cost_c.append(c)
        # self.mpc_cost_Ain.append(Ain)
        # self.mpc_cost_Bin.append(bin)
        # self.mpc_cost_Aeq.append(Aeq)
        # self.mpc_cost_Beq.append(beq)

if __name__ == '__main__':
    print("testing mucojo_robot.py setup")
    mpc = MPC()
    count = 0 
    while count<10:
        mpc.run_mpc()
        count= count+1
    print("all good so far")