import numpy as np
import matplotlib.pyplot as plt


class ControlPerformance():
    
    def __init__(self, controller_name):
        self.controller_name = controller_name
        self.control_input = []
        self.control_output = []
        self.position = []
        self.velocity = []
        self.target_position = []
        self.target_velocity = []
        self.force = []
        self.error =[]
        self.time = []

    def append_values(self, u, y, x, dx, target_x, target_dx, T, e, t):
        self.control_input.append(u)
        self.control_output.append(y)
        self.position.append(x)
        self.target_position.append(target_x)
        self.velocity.append(dx)
        self.target_velocity.append(target_dx)
        self.force.append(T)
        self.error.append(e)
        self.time.append(t)

    def convert_np(self):
        self.control_input_ = np.asarray(self.control_input)
        self.control_output_ = np.asarray(self.control_output)
        self.position_ = np.asarray(self.position)
        self.velocity_ = np.asarray(self.velocity)
        self.force_ = np.asarray(self.force)
        self.error_ = np.asarray(self.error)
        self.time_ = np.asarray(self.time)    
        self.target_position_ = np.asarray(self.target_position)    
        self.target_velocity_ = np.asarray(self.target_velocity)   

    def save_np_file(self):
        np.savez(
        self.controller_name + ".npz", 
        control_input=self.control_input_,
        control_output=self.control_output_,
        position=self.position_,
        velocity=self.velocity_,
        force=self.force_,
        error=self.error_,
        time=self.time_,
        target_position=self.target_position_,
        target_velocity=self.target_velocity_
        )

    def plot_performance(self):

        # Create a figure and a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))

        # First subplot: Control Input
        
        # for i in range(motor_u.shape[1]):
        for i in range(self.control_input_.shape[1]):
            axs[0, 0].plot(self.time_, self.control_input_[:,i],label=f'input {i}')
        axs[0, 0].set_title(' Input')
        axs[0, 0].legend()
        # Second subplot: Output
        for i in range(self.control_output_.shape[1]):
            axs[0, 1].plot(self.time_, self.control_output_[:,i], label=f'Output {i}')
            axs[0,1].plot(self.time_, self.target_position_[:,i], label=f'target_Position {i}')
        axs[0, 1].set_title('Output')
        axs[0, 1].legend()

        # Third subplot: Theta - Position
        for i in range(self.position_.shape[1]):
            axs[1, 0].plot(self.time_, self.position_[:,i], label=f'Position {i}')
        axs[1, 0].set_title('Position')
        axs[1, 0].legend()

        # Fourth subplot: D Theta - Velocity
        for i in range(self.velocity_.shape[1]):
            axs[1, 1].plot(self.time_, self.velocity_[:, i], label=f'Velocity {i}')
            # axs[1, 1].plot(self.time_, self.target_velocity_[:, i], label=f'target_Velocity {i}')
        axs[1, 1].set_title('Velocity')
        axs[1, 1].legend()

        for i in range(self.error_.shape[1]):
            axs[1, 2].plot(self.time_, self.error_[:, i], label=f' Error {i}')
        axs[1, 2].set_title('error')
        axs[1, 2].legend()
        
        for i in range(self.force_.shape[1]):
            axs[0, 2].plot(self.time_, self.force_[:, i], label=f'force motor {i}')
        axs[0, 2].set_title(' force motor')
        axs[0, 2].legend()


        # Adjust the layout and save the plot
        plt.tight_layout()
        plt.savefig(self.controller_name+'.png')    