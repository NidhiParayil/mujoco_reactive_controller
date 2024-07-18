from mjcf_urdf_simple_converter import convert

arm = "xarm7"
path2xml = "/home/nidhi/MPC_python/mujoco_menagerie/ufactory_xarm7/"+arm+".xml"
urdf = convert(path2xml,"/home/nidhi/MPC_python/mujoco_rbt/urdf/xarm7.urdf")