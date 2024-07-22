
try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

import time

import mujoco
import mujoco.viewer

from mujoco.glfw import glfw
from mpc_controller import MPC

from mujoco_env import MuJoCoBase
import numpy as np

import roboticstoolbox as rtb
import spatialmath.base as base

class RoboEnv(MuJoCoBase):

    def __init__(self, path2urdf="./urdf/xarm7.urdf"):
        # xml_path = "/home/nidhi/MPC_python/mujoco_menagerie/ufactory_xarm7/xarm7.xml"
        # xml_path = "/home/nidhi/MPC_python/mujoco_rbt/assets/interface.xml"
        xml_path = 'C:/Users/NidhiParayil/nidhi/mujoco_reactive_controller/assets/interface.xml'
        super().__init__(xml_path)
        
        self.data = mujoco.MjData(self.model)
        # self.model.from_xml_path("home/nidhi/MPC_python/mujoco_rbt/assets=/ball.xml") 
        self.start_time = time.time
        self.joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
        self.joint_ids = self.get_jointIds()
        self.max_position = [3,3,3,3,3,3,3]
        self.min_position = [-3,-3,-3,-3,-3,-3,-3]
        self.gain_parm = [1500, 1500, 1000, 1000, 1000, 800, 800]
        self.min_velocity=[-10, -10, -10, -10, -10, -10,-10]
        self.max_velocity=[10, 10, 10, 1, 10, 10,10]
        self.min_effort = np.array([-2000,-2000,-2000,-2000,-2000,-2000,-2000])
        self.max_effort = np.array([2000,2000,2000,2000,2000,2000,2000])
        self.antiwindup = False
        self.reset_joints()
        self.integral = np.zeros(7)
        self.prev_error = np.zeros(7)
        # robot setup
        self.robot = rtb.robot.Robot.URDF(file_path=path2urdf)
        self.link1 = rtb.robot.DHLink(d=0.267, alpha=0.0, theta=0.0, a=0.0, m =2.382)
        self.link2 = rtb.robot.DHLink(d=0., alpha=-np.pi/2, theta=0.0, a=0.0, m =1.869)
        self.link3 = rtb.robot.DHLink(d=0.293, alpha=np.pi/2, theta=0.0, a=0.0, m = 1.6383)
        self.link4 = rtb.robot.DHLink(d=0., alpha=-np.pi/2, theta=0.0, a=0.0525, m = 1.7269)
        self.link5 = rtb.robot.DHLink(d=0.3425, alpha=np.pi/2, theta=0.0, a=0.0775, m =1.3203)
        self.link6 = rtb.robot.DHLink(d=0., alpha=np.pi/2, theta=0.0, a=0.0, m= 1.325)
        self.link7 = rtb.robot.DHLink(d=0.097, alpha=-np.pi/2, theta=0.0, a=0.076, m = 0.17)
        self.robot_dh = rtb.robot.DHRobot([self.link1, self.link2,self.link3, self.link4,self.link5, self.link6, self.link7])
        self.start_time = time.time()
        self.curr_time = time.time() - self.start_time
        print("---------all good loading robots------------")

    def reset_joints(self):

        angles = [0,-1, 0, 0, 0.0, np.pi/3, 0]
        for joint, ang in zip(self.joint_ids, angles):
            self.data.qpos[joint] = ang



        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.run(angles)
        time.sleep(2)
        print("robot set to initial pose")


    def get_jointIds(self):
        joint_ids =[]
        for joint in self.joint_names:
            joint_ids.append(self.model.jnt(joint).id)
        return joint_ids

    def get_joint_positions(self):
        q = []
        for j in self.joint_ids:
            q.append(self.data.qpos[j])
        return q

    def get_joint_vel(self):
        dq = []
        for j in self.joint_ids:
            dq.append(self.data.qvel[j])

        return dq
    
    def get_joint_acc(self):
        ddq = []
        for j in self.joint_ids:
            ddq.append(self.data.qacc[j])

        return ddq        

    def get_ee_wrench(self):
        # print(self.model.name_sensoradr)
        # self.force_id = self.model.name_sensoradr[0]
        # self.force_adr = self.model.name_sensoradr[0]
        # self.torque_id = self.model.name_sensoradr[1]
        # # self.torque_adr = self.model.name_sensoradr[1]
        # self.sensor_site = "eef_site "
        # # print(self.data.site_xmat.shape)
        # self.ft_site_id = self.model.name_siteadr[1]
        self.force_ndim = 3
        self.torque_ndim = 3
        # Get the orientation matrix of the force-torque (FT) sensor
        ft_ori_mat = self.data.site_xmat[1,:].reshape(3, 3)
        force = self.data.sensordata[0:3]
        torque = self.data.sensordata[3:6]
        # print(self.data.sensordata)
        force = ft_ori_mat @ force
        torque = ft_ori_mat @ torque
        wrench = np.concatenate([force, torque])
        # print("t", torque)
        wrench = np.array([wrench[1],wrench[2],wrench[0],wrench[4],wrench[5],wrench[4]])
        return wrench

    def get_ee_position(self):
        body_name = "link7"
        return self.data.body(body_name).xpos

    def get_ee_pose(self):
        return self.data.xpos

    def get_jacobian(self):
        q_mujoco = self.get_joint_positions()
        J = np.array(self.robot.jacobe(q_mujoco))
        return J
    


    def get_rbt_joint_pos(self):
        pass

    def get_rbt_joint_vel(self):
        pass


    def get_forward_dynamics(self, wrench):
        joint_torque= self.data.qfrc_actuator[self.joint_ids]
        q = self.get_joint_positions()
        dq = self.get_joint_vel()
        M_q = self.get_M_()
        c_q = self.data.qfrc_bias[self.joint_ids]
        J_q = self.get_jacobian()

        Q=np.matmul(np.linalg.inv(M_q), (joint_torque  - np.dot(J_q.T, wrench)))    
        ddq = []
        for joint in self.joint_ids:
            ddq.append(Q[joint])
        return ddq

    def get_inverse_dynamics(self):
        joint_torque= self.data.qfrc_actuator[self.joint_ids]
        ddq = self.get_joint_acc()
        q = self.get_joint_positions()
        dq = self.get_joint_vel()
        M_q = self.get_M_()
        c_q = self.data.qfrc_bias[self.joint_ids]
        J_q = self.get_jacobian()
        T = joint_torque* q
        # Q=np.matmul(np.linalg.inv(M_q), (joint_torque  - np.dot(J_q.T, wrench)))    
        # ddq = []
        # for joint in self.joint_ids:
        #     ddq.append(Q[joint])
        return T


    def get_rbt_end_eff_pose(self):
        q_mujoco = self.get_joint_positions()
        pose = self.robot.fkine(q_mujoco)
        pos = pose.t
        orient = pose.R
        return pose, pos, orient

    def get_M_(self):
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        M = M_full[self.joint_ids, :][:, self.joint_ids]
        return M
    def update_sim(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
        glfw.poll_events()




    # def joint_pos_controller(self, target_joint_position):
    #     M = self.get_M_()
        
    #     # print(target_joint_position)
    #     target_positions = np.clip(target_joint_position[0:7], self.min_position, self.max_position)
    #     current_positions = self.data.qpos[self.joint_ids]
    #     current_velocities = self.data.qvel[self.joint_ids]

    #     # Calculate the error term for PD control
    #     position_error = target_positions - current_positions
    #     velocity_error = -current_velocities
    #     self.kp=[x * 0 for x in self.gain_parm]
    #     self.kd=[x * 0 for x in self.gain_parm]
    #     # Calculate the control effort (u) using PD control
    #     # print()
    #     u = self.kp * position_error + self.kd * velocity_error + target_joint_position

    #     # Apply the joint space inertia matrix to obtain the desired joint effort (torques)
    #     # target_effort = np.dot(M, u)
    #     self.run(u)


    # def joint_velocity_controller(self, target_joint_velocity):
    #     M = self.get_M_()
    #     # M = M_full[self.joint_ids, :][:, self.joint_ids]
    #     target_velocities = np.clip(target_joint_velocity, self.min_velocity, self.max_velocity)
    #     current_velocities = self.data.qvel[self.joint_ids]
    #     error = target_velocities - current_velocities

    #     # Integral term
    #     if self.antiwindup:
    #         self.integral += error
    #         # Anti-windup to limit the integral term
    #         self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
    #     else:
    #         self.integral += error

    #     self.kp=[x *.01 for x in self.gain_parm]
    #     self.kd=[x *.001 for x in self.gain_parm]
    #     self.ki=[x *(-0.0) for x in self.gain_parm]
    #     # Calculate the control effort (u) using PD control
    #     # Derivative term
    #     derivative = error - self.prev_error

    #     # Calculate the desired joint effort (torques) using PID control
    #     u = self.kp * error + self.kd * derivative + self.ki * self.integral 
    #     # Save the current error for the next iteration
    #     self.prev_error = error
    #     target_effort = np.dot(M, u)
    #     self.run(target_effort, u_motor)

    # # def joint_effort_controller(self, target):

    #     target_effort = np.clip(target, self.min_effort, self.max_effort)
        
    #     # Get the gravity compensation for each joint
    #     gravity_compensation = self.data.qfrc_bias[self.joint_ids]

    #     # Add gravity compensation to the target effort
    #     target_effort += gravity_compensation

    #     # Call the base controller's `run_controller` method with the clipped target efforts
    #     self.run(target_effort, 5)


    def run(self, control_input):
        # this works like a position controller 
        self.data.ctrl[0:7] = control_input[0:7]
        self.data.ctrl[-1] = 5
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self.update_sim()


if __name__ == '__main__':
    print("testing robot.py setup")
    env = RoboEnv()
    count = 0 
    mpc = MPC()
    q= env.get_joint_positions()
    dq = env.get_joint_vel()
    rx = .8
    ry =-.2
    rz = .3
    while count< 100:
        mpc.run_mpc(rx,ry,rz,q, env.joint_ids,dq, env)
        count = count+1
    print("all good so far")