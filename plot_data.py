import numpy as np 
import matplotlib.pyplot as plt
import time


def generate_save_plt(dx, dy, x, y, title, axis_names, name, legend, time, time1, time2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    colors = ["green","blue","red", "yellow", "orange","pink"]
    # for i in range(0,y.shape[1]):
    plt.plot(dx,dy,colors[1])
    plt.plot(x,y,colors[0])
    
    plt.title(title)
    plt.legend(legend)
    # plt.axis("equal")
    ax.set_xlim(time1, time2)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/'+title+name+'.png')

def generate_save_3plt(time_, x, y, z, title, axis_names, name, legend, time, time1, time2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    colors = ["green","blue","red", "yellow", "orange","pink"]
    # for i in range(0,y.shape[1]):
    plt.plot(time_,y,colors[0])
    plt.plot(time_,z,colors[2])
    plt.plot(time_,x,colors[1])
    
    plt.title(title)
    plt.legend(legend)
    # plt.axis("equal")
    ax.set_xlim(time1, time2)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/'+title+name+'.png')




def generate_save_joint_plt(x, y, title, axis_names,name, legend, time, time1, time2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    colors = ["green","blue","red", "yellow", "orange","pink","black"]
    # for i in range(0,y.shape[1]):

    plt.plot(x,y[:,1],colors[0])
    plt.plot(x,y[:,2],colors[1])
    plt.plot(x,y[:,3],colors[2])
    plt.plot(x,y[:,4],colors[3])
    plt.plot(x,y[:,5],colors[4])
    plt.plot(x,y[:,6],colors[5])
    plt.plot(x,y[:,0],colors[6])

    ax.set_xlim(time1, time2)
    plt.title(title)
    plt.legend(legend)
    # plt.axis("equal")
    # ax.set_xlim(-.6, .8)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/'+title+name+'.png')    



def generate_save_mpc_plt(x, y, title, axis_names,name, legend, time, time1, time2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    x =  np.arange(0, y.shape[0])
    colors = ["green","blue","red", "yellow", "orange","pink","black"]
    # for i in range(0,y.shape[1]):
    plt.plot(x,y[:,1],colors[0])
    plt.plot(x,y[:,2],colors[1])
    plt.plot(x,y[:,3],colors[2])
    plt.plot(x,y[:,4],colors[3])
    plt.plot(x,y[:,5],colors[4])
    # plt.plot(x,y[:,6],colors[5])
    # plt.plot(x,y[:,7],colors[6])

    ax.set_xlim(time1, time2)
    plt.title(title)
    plt.legend(legend)
    # plt.axis("equal")
    # ax.set_xlim(-.6, .8)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/'+title+name+'.png')    

def generate_save_mpc_2dplt(x, y, title, axis_names,name, legend, time, time1, time2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    x =  np.arange(0, y.shape[0])
    colors = ["green","blue","red", "yellow", "orange","pink","black"]
    # for i in range(0,y.shape[1]):
    plt.plot(x,y[:,1,1],colors[0])
    plt.plot(x,y[:,2,2],colors[1])
    plt.plot(x,y[:,3,3],colors[2])
    plt.plot(x,y[:,4,4],colors[3])
    plt.plot(x,y[:,5,5],colors[4])
    plt.plot(x,y[:,6,6],colors[5])
    plt.plot(x,y[:,0,0],colors[6])

    ax.set_xlim(time1, time2)
    plt.title(title)
    plt.legend(legend)
    # plt.axis("equal")
    # ax.set_xlim(-.6, .8)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.savefig('./plot_results/'+title+name+'.png')    



def generate_save_obj_plt(x, y, title, axis_names,name, legend, time, time1, time2):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    # plt.grid(True)
    x =  np.arange(0, y.shape[0])
    colors = ["green","blue","red", "yellow", "orange","pink","black"]
    # for i in range(0,y.shape[1]):
    ax[0].plot(x,y[:,0],colors[0])
    ax[1].plot(x,y[:,1],colors[1])
    ax[2].plot(x,y[:,2],colors[2])

    ax[0].set_xlim(time1, time2)
    # ax[0].title(title)
    # ax.legend(legend)
    # plt.axis("equal")
    # ax.set_xlim(-.6, .8)
    # ax.set_ylim(-.6,.8)
    # ax.set_zlim(-.5, .5)
    ax[2].set_xlabel("iteration")
    ax[0].set_ylabel(legend[0])
    ax[1].set_ylabel(legend[1])
    ax[2].set_ylabel(legend[2])
    fig.suptitle("mpc objectve function")
    plt.savefig('./plot_results/'+title+name+'.png')    


name = 'pid_position'
data = np.load(name +'.npz')
robo_x = data['robo_x']
robo_y = data['robo_y']
robo_z = data['robo_z']
# robo_fn = data['robo_fx']
# robo_fy = data['robo_fy']
# robo_fz = data['robo_fz']
time_= data['time_']
robo_q = data['robo_q']
robo_dq = data['robo_dq']

err_x = data["err_x"]
err_y = data["err_y"]
err_z = data["err_z"]
# robo_ctl = data['robo_ctl']  


# mpc_ctl = data['mpc_ctl'] 
# mpc_cost_Q = data['mpc_cost_Q']
# mpc_cost_c = data['mpc_cost_c']
# mpc_cost_Aeq = data['mpc_cost_Aeq']
# mpc_cost_Beq = data['mpc_cost_Beq']
# mpc_cost_Ain = data['mpc_cost_Ain']
# mpc_cost_Bin = data['mpc_cost_Bin']
# mpc_obj = data["mpc_obj"]

x_des = np.ones(robo_x.shape)*.7
y_des = np.ones(robo_x.shape)*(-.33)
z_des = np.ones(robo_x.shape)*.322


generate_save_plt(time_,x_des, time_, robo_x,  title =" x"+name, axis_names =["time", "position x"],  name = name, legend=["des", "robot"], time=True, time1=time_[0],time2= time_[-1])
generate_save_plt(time_,y_des, time_, robo_y, title ="y "+name, axis_names =["time", "position y"],  name = name, legend=["des", "robot"], time=True, time1=time_[0],time2= time_[-1])
generate_save_plt(time_,z_des, time_, robo_z, title ="z "+name, axis_names =["time", "position z"],  name = name, legend=["des", "robot"], time=True, time1=time_[0],time2= time_[-1])
# generate_save_plt(time_,robo_fn,time_, robo_fn,  title ="wrench ", axis_names =["time", "force norm"], name = name, legend=["x", "x"], time=True, time1=time_[0],time2= time_[-1])




generate_save_joint_plt(time_,robo_q, title ="joint pos ", axis_names =["time", "angle"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=time_[0],time2= time_[-1])
generate_save_joint_plt(time_,robo_dq, title ="joint vel ", axis_names =["time", "angle"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=time_[0],time2= time_[-1])
# generate_save_joint_plt(time_,robo_ctl, title ="joint ctl "+name, axis_names =["time", "angle"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=time_[0],time2= time_[-1])
# generate_save_joint_plt(time_,mpc_ctl, title ="mpc ctl ", axis_names =["time", "angle"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=time_[0],time2= time_[-1])


# generate_save_mpc_2dplt(time_,mpc_cost_Q,  title ="mpc_cost_Q ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_Q.shape[0])
# print(mpc_obj)
# generate_save_mpc_plt(time_,mpc_cost_c,  title ="mpc cost_c ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_c.shape[0])
# generate_save_mpc_plt(time_,mpc_cost_Aeq,  title ="mpc A eq ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_Aeq.shape[0])
# generate_save_mpc_plt(time_,mpc_cost_Ain,  title ="mpc A in ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_Ain.shape[0])
# generate_save_mpc_plt(time_,mpc_cost_Beq,  title ="mpc b eq ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_Beq.shape[0])
# generate_save_mpc_plt(time_,mpc_cost_Bin,  title ="mpc b in ", axis_names =["time", "cost"], name = name, legend=["1", "2","3","4","5","6","7"], time=True, time1=0,time2= mpc_cost_Bin.shape[0])
# generate_save_obj_plt(time_,mpc_obj,  title ="mpc objective funtion ", axis_names =["time", "cost"], name = name, legend=["total","term1", "term2"], time=True, time1=0,time2= mpc_cost_Bin.shape[0])



generate_save_3plt(time_,err_x, err_y, err_z, title =" errorz"+name, axis_names =["time", "position error"],  name = name, legend=["x", "y", "z"], time=True, time1=time_[0],time2= time_[-1])


print("all good so far")