
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
# Variables : velocity
# Objective : velocity deviation
# Constraints : Ax +Bu 

class MPC():

    def __init__(self, dt):
        cvxopt.solvers.options['show_progress'] = False
        # MPC Parameters
        self.n = 14  # Number of joints
        self.N = 10  # MPC prediction horizon (20 to 30 samples)
        self.m = 1 # control horizon, (.1 * prediction) 
        self.dt = dt
        # self.tmax = tmax
        # MPC setup
        self.q0 = np.zeros(self.n)

        self.A = np.eye(self.n)
        self.B = self.dt * np.eye(self.n)
        self.Q = 1e8 * np.eye(self.n)
        self.Ru = np.eye(self.n)
        self.P = self.Q
        self.S, self.M, self.Qbar,self. Rbar, _ = uncMPC(self.N, self.A, self.B, self.Q, self.Ru, self.P)
        self.H = 2 * np.dot(self.S.T, np.dot(self.Qbar, self.S)) + self.Rbar
        self.f0 = 2 * np.dot(self.S.T, np.dot(self.Qbar, self.M))
        # print("forceeeeee", self.f0.shape)

        # robot setup
        # self.robot = rtb.robot.Robot.URDF(file_path=path2urdf)

        # Robot-specific Constraints
        self.u_UB = np.array([10, 10, 10,10, 10, 10, 10,10, 10, 10,10, 10, 10, 10])*10 # Max speed (degree/s)
        self.u_LB = -self.u_UB
        self.U_UB = np.tile(self.u_UB, self.N)
        self.U_LB = np.tile(self.u_LB, self.N)
        self.q_UB = np.array([6.28319, 6.28319, 2.61799, 6.28319, 6.28319, 6.28319,6.28319,1000,1000,1000,1000,1000,1000,1000]) # Motion range (rad)
        self.q_LB = -self.q_UB
        self.Q_UB = np.tile(self.q_UB, self.N)
        self.Q_LB = np.tile(self.q_LB, self.N)
        self.G = np.vstack((self.S, -self.S, np.eye(self.S.shape[1]), -np.eye(self.S.shape[1])))
        self.W = np.hstack((self.Q_UB, -self.Q_LB, self.U_UB, -self.U_LB))
        self.T = np.vstack((-self.M, self.M, np.zeros_like(self.M), np.zeros_like(self.M)))
        # self.Wtil = (self.W + np.dot(self.T, self.q0)).reshape(-1,1)
        # print("w", self.Wtil.shape, self.G.shape)
        self.U_UB = self.U_UB.reshape(-1,1)
        self.U_LB = self.U_LB.reshape(-1,1)
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


    def quadprog(self, H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        n_var = H.shape[1]

        P = cvxopt.matrix(H, tc='d')
        q = cvxopt.matrix(f, tc='d')

        if L is not None or k is not None:
            assert(k is not None and L is not None)
            if lb is not None:
                L = np.vstack([L, -np.eye(n_var)])
                k = np.vstack([k, -lb])

            if ub is not None:
                L = np.vstack([L, np.eye(n_var)])
                k = np.vstack([k, ub])

            L = cvxopt.matrix(L, tc='d')
            k = cvxopt.matrix(k, tc='d')

        if Aeq is not None or beq is not None:
            assert(Aeq is not None and beq is not None)
            Aeq = cvxopt.matrix(Aeq, tc='d')
            beq = cvxopt.matrix(beq, tc='d')
        sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

        return np.array(sol['x'])


    def run_opt_controller(self,target_position, target_vel, q,dq, robot):
        val , pos, oreint =   robot.get_rbt_end_eff_pose()
        Tep = val
        Tep.t = target_position
        n = self.n
        pg = target_position
        Rg = oreint
        self.Tg = SE3.Trans(pg) * Rg
        self.q = q
        self.q0 = q
        i = 0
        self.pre_error =10
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

            self.Q = np.eye(n)
            self.Q[:n, :n] *= self.Y

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
            Aeq = np.zeros((12,n))
            beq = np.zeros(12)    
            # print(Aeq)        
            Aeq[0:6,0:7] = robot.get_jacobian()
            Aeq[6:12,7:14] = robot.get_jacobian()
            # print(vel_k_1)
            beq[0:6]= np.asarray(target_vel) 
            Torque = robot.get_inverse_dynamics()
            
            force_ex = np.round(wrench - np.dot(J, Torque), 1)
            beq[6:12]=force_ex
            print("force", force_ex)
            J_inv = np.linalg.pinv(J)
            w_tor = np.dot(J_inv, wrench)
            print("cdsfdfd")
            print(w_tor)
            print("ttttt")
    
            # print(beq[0], target_vel[0])
            # print("pos x, beq", robot.data.site_xpos[0][0], np.dot(J,dq))


            # print("w",  wrench)
            # print("T", np.dot(J, Torque))
            # time.sleep(5)
            # The inequality constraints for joint limit avoidance
            Ain = np.zeros((n, n))
            bin = np.zeros(n)

            # The minimum angle (in radians) in which the joint is allowed to approach
            # to its limit
            ps = 0.05

            # The influence angle (in radians) in which the velocity damper
            # becomes active
            pi = 0.9

            # Form the joint limit velocity damper
            Ain[:7, :7], bin[:7] = robot.robot.joint_velocity_damper(ps, pi, 7)
                # Linear component of objective function: the manipulability Jacobian
            # c = np.r_[-np.delete(np.array(robot.robot.jacobe(robot.robot.q)), -1, axis = 1).reshape((n,)), np.zeros(6)]

            # The lower and upper bounds on the joint velocity and slack variable
            lb = self.q_LB
            ub = self.q_UB
            
            
            # c = np.matmul(np.transpose(J), wrench)*1000
    
            c = np.zeros(n)
            dq_d = qp.solve_qp(self.Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver = 'cvxopt')  
  
            q= robot.get_joint_positions()  
            ee_pos = robot.get_ee_position()   
            
            if dq_d is None:
                u = self.prev_u[0:7]
                robot.run(u[:7])
                self.mpc_ctl.append(self.prev_u)
            else:
                # qd[5] = 0
                # qd[6] = 0
                # qd[7] = 0
                u = dq_d[0:7]*dt*60 #+ np.dot(M_inv,dq_d[7:14])*1200*dt*dt/2
                robot.run(u[:7])
                self.prev_u =u[:n]
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
            u = dq_d[0:7]*dt*60+np.dot(M_inv,dq_d[7:14])*1200*dt*dt/2
            print(dt)
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