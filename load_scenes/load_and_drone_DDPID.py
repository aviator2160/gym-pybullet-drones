"""Simulation script for assignment 1.

The script uses the control defined in file `aer1216_fall2020_hw1_ctrl.py`.

Example
-------
To run the simulation, type in a terminal:

    $ python aer1216_fall2020_hw1_sim.py

"""
import time
import random
import numpy as np
import pybullet as p

#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
import sys

sys.path.append('./')
sys.path.append('../')

import os
print(os.path.dirname)

from gym_pybullet_drones.envs.BaseAviary import PayloadObject, jointFactory
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.control.ForcePIDControl import ForcePIDControl
from gym_pybullet_drones.control.DDLoadControl import DDLoadControl

DURATION = 10
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""

NUM_DRONES = 1
NUM_LOADS = 1

# INIT_XYZS = np.array([[ 0.25,     0, 0.5],
#                       [    0,  0.25, 0.5],
#                       [-0.25,     0, 0.5],
#                       [    0, -0.25, 0.5],
#                       [    0,     0, 0.1],])

# LOADS = [PayloadObject(mass=1,
#                        dims=np.array([0.2, 0.2, 0.2]),
#                        hardpoints=np.array([[0.25, 0.25, 0.25], [0.25, -0.25, 0.25], [-0.25, 0.25, 0.25], [-0.25, -0.25, 0.25]])
#                       ),]

# JOINTS = jointFactory(joints=[[0,0,0],
#                               [0,0,1],
#                               [0,0,2],
#                               [0,0,3],
#                              ],
#                       num_drones=NUM_DRONES, init_xyzs=INIT_XYZS, load_desc=LOADS)


INIT_XYZS = np.array([[ 0,  0,   1],
                      [   0,    0, 0.2],])

LOADS = [PayloadObject(mass=0.002,
                       dims=np.array([0.2, 0.2, 0.1]),
                       hardpoints=np.array([[0,    0, 0.05],
                                            ]),
                      ),]

JOINTS = jointFactory(joints=[[0,0,0],
                             ],
                      num_drones=NUM_DRONES, init_xyzs=INIT_XYZS, load_desc=LOADS)

if __name__ == "__main__":

    #### Create the ENVironment ################################
    ENV = CtrlAviary(num_drones=NUM_DRONES, num_loads=NUM_LOADS, initial_xyzs=INIT_XYZS, loads=LOADS, joints=JOINTS, gui=GUI)
    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ)

    #### Initialize the controller #############################
    DRONE_CTRL = ForcePIDControl()

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()

    LOAD_CTRL = DDLoadControl(LOADS[0], doNotStabilize=True)
    # print(LOAD_CTRL.F)

    # LOAD_STR = str(NUM_DRONES)
    # STATE = OBS[LOAD_STR]["state"]
    prev_drone_vels = np.zeros((3, NUM_DRONES))
    # forces = LOAD_CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
    #                                   cur_pos=STATE[0:3],
    #                                   cur_quat=STATE[3:7],
    #                                   cur_vel=STATE[10:13],
    #                                   cur_ang_vel=STATE[13:16],
    #                                   target_pos=STATE[0:3]
    #                                  )
    for i in range(NUM_DRONES):
        STATE = OBS[str(i)]["state"]
        prev_drone_vels[:,i] = STATE[10:13]
        rpm = DRONE_CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
                                        cur_quat=STATE[3:7],
                                        cur_vel=STATE[10:13],
                                        cur_accel=np.zeros(3,),
                                        cur_ang_vel=STATE[13:16],
                                        target_force=np.array([0.001, 0, 0.002 * 9.81])
                                       )
        ACTION[str(i)] = rpm

    #### Initialize target trajectory ##########################
    TARGET_POSITION = np.array([[0, 0, 1.0] for i in range(DURATION*ENV.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * ENV.SIM_FREQ, 3])

    #### Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / ENV.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / ENV.SIM_FREQ

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, DURATION*ENV.SIM_FREQ):

        ### Secret control performance booster #####################
        # if i/ENV.SIM_FREQ>3 and i%30==0 and i/ENV.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        OBS, _, _, _ = ENV.step(ACTION)

        #### Compute control #######################################
        # STATE = OBS[LOAD_STR]["state"]
        # forces = LOAD_CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
        #                                   cur_pos=STATE[0:3],
        #                                   cur_quat=STATE[3:7],
        #                                   cur_vel=STATE[10:13],
        #                                   cur_ang_vel=STATE[13:16],
        #                                   target_pos=STATE[0:3]
        #                                  )
        for i in range(NUM_DRONES):
            STATE = OBS[str(i)]["state"]
            cur_vel = STATE[10:13]
            cur_accel = (cur_vel - prev_drone_vels[:,i])/ENV.SIM_FREQ
            prev_drone_vels[:,i] = cur_vel
            rpm = DRONE_CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
                                            cur_quat=STATE[3:7],
                                            cur_vel=STATE[10:13],
                                            cur_accel=cur_accel,
                                            cur_ang_vel=STATE[13:16],
                                            target_force=np.array([0.001, 0, 0.002 * 9.81])
                                           )
        ACTION[str(i)] = rpm

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    LOGGER.save()

    #### Plot the simulation results ###########################
    LOGGER.plot()
