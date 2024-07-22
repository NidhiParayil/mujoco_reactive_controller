# mujoco_reactive_controller


Setting up code:

run the requirements

OR run the following commands to manually install 
(curently - june 2024 - mujoco works with python 3.11)
numpy version used is 1.23.5


py -3.11 -m pip install mujoco   
py -3.11 -m pip install cvxopt  
py -3.11 -m pip install urdfpy 
py -3.11 -m pip install roboticstoolbox-python
py -3.11 -m pip install qpsolvers

(windows)

Python versions are slightly conflicting i think
- modified "from collections" to "from collections.abc"
- modified "from fraction import gcd" to "from math import gcd"
- 