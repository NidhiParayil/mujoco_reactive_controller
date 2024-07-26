import sys
import time
import numpy as np
from dm_control.mujoco.wrapper import mjbindings
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import roboticstoolbox as rtb
import spatialmath.base as base
from mpc_controller import MPC
from mujoco_env import MuJoCoBase

class RoboEnv(MuJoCoBase):
    def __init__(self):
        is_windows = sys.platform.startswith('win')
        xml_path, urdf_path = self.get_paths(is_windows)
        super().__init__(xml_path)
        self.data = mujoco.MjData(self.model)
        self.start_time = time.time()
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_ids = self.get_joint_ids()
        self.set_robot_params()
        self.robot_rtb = self.load_rtb_robot(urdf_path)
        self.reset_joints()
        print("---------all good loading robots------------")

    def get_paths(self, is_windows):
        if is_windows:
            xml_path = 'C:/Users/NidhiParayil/nidhi/mujoco_reactive_controller/assets/interface.xml'
            urdf_path = "./urdf/xarm7.urdf"
        else:
            xml_path = "/home/nidhi/MPC_python/mujoco_reactive_controller/assets/interface.xml"
            urdf_path = "/home/nidhi/MPC_python/mujoco_reactive_controller/urdf/xarm7.urdf"
        return xml_path, urdf_path

    def set_robot_params(self):
        self.max_position = [3, 3, 3, 3, 3, 3, 3]
        self.min_position = [-3, -3, -3, -3, -3, -3, -3]
        self.gain_parm = [1500, 1500, 1000, 1000, 1000, 800, 800]
        self.min_velocity = [-10, -10, -10, -10, -10, -10, -10]
        self.max_velocity = [10, 10, 10, 10, 10, 10, 10]
        self.min_effort = np.array([-2000, -2000, -2000, -2000, -2000, -2000, -2000])
        self.max_effort = np.array([2000, 2000, 2000, 2000, 2000, 2000, 2000])
        self.antiwindup = False
        self.integral = np.zeros(7)
        self.prev_error = np.zeros(7)
        self.eef_name = "end_effector_dummy"
        self.eef_site_id = 7

    def load_rtb_robot(self, urdf_path):
        robot = rtb.robot.Robot.URDF(file_path=urdf_path)
        self.robot_dh = rtb.robot.DHRobot([
            rtb.robot.DHLink(d=0.267, alpha=0.0, theta=0.0, a=0.0, m=2.382),
            rtb.robot.DHLink(d=0., alpha=-np.pi/2, theta=0.0, a=0.0, m=1.869),
            rtb.robot.DHLink(d=0.293, alpha=np.pi/2, theta=0.0, a=0.0, m=1.6383),
            rtb.robot.DHLink(d=0., alpha=-np.pi/2, theta=0.0, a=0.0525, m=1.7269),
            rtb.robot.DHLink(d=0.3425, alpha=np.pi/2, theta=0.0, a=0.0775, m=1.3203),
            rtb.robot.DHLink(d=0., alpha=np.pi/2, theta=0.0, a=0.0, m=1.325),
            rtb.robot.DHLink(d=0.097, alpha=-np.pi/2, theta=0.0, a=0.076, m=0.17),
        ])
        return robot

    def reset_joints(self):
        angles = [0, -1, 0, 0, 0.0, np.pi/3, 0]
        for joint, ang in zip(self.joint_ids, angles):
            self.data.qpos[joint] = ang
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.run(angles)
        time.sleep(2)
        print("robot set to initial pose")



    #################################
    # sensor reading
    ################################        

    def get_ee_wrench(self):
        ft_ori_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        force = np.dot(ft_ori_mat, self.data.sensordata[21:24])
        torque = np.dot(ft_ori_mat, self.data.sensordata[24:27])
        return -np.concatenate([force, torque])

    def get_ee_vel(self):
        ft_ori_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        return np.dot(ft_ori_mat, self.data.sensordata[27:30])

    def get_ee_position(self):
        return self.data.body(self.eef_name).xpos

    def get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mjbindings.mjlib.mj_jacSite(self.model, self.data, jacp, jacr, self.eef_site_id)
        jac = np.vstack([jacp, jacr])
        return jac[:, self.joint_ids]

    def get_joint_torque_sensor(self):
        joint_torque_sensor = []
        for i in range(0, 7):
            ori_mat = self.data.site_xmat[i].reshape(3, 3)
            # torque = np.dot(ft_ori_mat, self.data.sensordata[3:6])
            torque =  self.data.sensordata[i*3:i*3+3]
            joint_torque_sensor.append(torque[np.argmax(np.abs(torque))])
        return joint_torque_sensor

    #################################
    # numerical modelling mujoco
    ################################
    def get_joint_ids(self):
        return [self.model.jnt(joint).id for joint in self.joint_names]

    def get_joint_positions(self):
        return [self.data.qpos[j] for j in self.joint_ids]

    def get_joint_vel(self):
        return [self.data.qvel[j] for j in self.joint_ids]

    def get_joint_acc(self):
        return [self.data.qacc[j] for j in self.joint_ids]

    def get_M_(self):
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        return M_full[self.joint_ids, :][:, self.joint_ids]

    def get_forward_dynamics(self, wrench):
        M_q = self.get_M_()
        Q = np.matmul(np.linalg.inv(M_q), self.data.actuator_force[self.joint_ids])
        return [Q[joint] for joint in self.joint_ids]

    def get_joint_torque_mujoco(self):
        joint_torque = self.data.qfrc_actuator[self.joint_ids]
        M = self.get_M_()
        ddq = self.get_joint_acc()
        c_q = self.data.qfrc_bias[self.joint_ids]
        T = np.dot(M, ddq) + c_q 
        return T, joint_torque

    #################################
    #           rbt
    ################################

    def get_rtb_end_eff_pose(self):
        q_mujoco = self.get_joint_positions()
        pose = self.robot.fkine(q_mujoco)
        return pose, pose.t, pose.R

    def get_rtb_jacobian(self)

    def get_rtb_joint_torque(self):
        pass





    #################################
    #           run sim
    ################################


    def update_sim(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def run(self, control_input):
        self.data.ctrl[0:7] = control_input
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self.curr_time = time.time() - self.start_time
        self.update_sim()


if __name__ == '__main__':
    print("Testing robot.py setup")
    env = RoboEnv()
    mpc = MPC(dt=1/60)  # Assuming dt is 1/60
    count = 0
    q = env.get_joint_positions()
    dq = env.get_joint_vel()
    rx, ry, rz = 0.8, -0.2, 0.3
    while count < 100:
        mpc.run_opt_controller(target_position=[rx, ry, rz], target_vel=[0, 0, 0], q=q, dq=dq, robot=env)
        count += 1
    print("All good so far")