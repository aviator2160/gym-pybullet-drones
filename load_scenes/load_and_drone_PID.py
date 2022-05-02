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
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

DURATION = 10
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""

NUM_DRONES = 4
NUM_LOADS = 1

INIT_XYZS = np.array([[ 0.1,    0,   1],
                      [   0,  0.1,   1],
                      [-0.1,    0,   1],
                      [   0, -0.1,   1],
                      [   0,    0, 0.9],])

LOADS = [PayloadObject(mass=0.01,
                       dims=np.array([0.2, 0.2, 0.1]),
                       hardpoints=np.array([[ 0.1,    0, 0.05],
                                            [   0,  0.1, 0.05],
                                            [-0.1,    0, 0.05],
                                            [   0, -0.1, 0.05],]),
                      ),]

JOINTS = jointFactory(joints=[[0,0,0],
                              [1,0,1],
                              [2,0,2],
                              [3,0,3],],
                      num_drones=NUM_DRONES, init_xyzs=INIT_XYZS, load_desc=LOADS)

if __name__ == "__main__":

    #### Create the ENVironment ################################
    ENV = CtrlAviary(num_drones=NUM_DRONES, num_loads=NUM_LOADS, initial_xyzs=INIT_XYZS, loads=LOADS, joints=JOINTS, gui=GUI)
    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ)

    #### Initialize the controller #############################
    CTRL = DSLPIDControl()

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()
    STATE = OBS["0"]["state"]
    rpm, _, _ = CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
                                         cur_pos=STATE[0:3],
                                         cur_quat=STATE[3:7],
                                         cur_vel=STATE[10:13],
                                         cur_ang_vel=STATE[13:16],
                                         target_pos=STATE[0:3]
                                         )
    ACTION["0"] = rpm

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
        STATE = OBS["0"]["state"]
        rpm, _, _ = CTRL.computeControl(control_timestep=ENV.SIM_FREQ, # TODO: control timestep
                                         cur_pos=STATE[0:3],
                                         cur_quat=STATE[3:7],
                                         cur_vel=STATE[10:13],
                                         cur_ang_vel=STATE[13:16],
                                         target_pos=TARGET_POSITION[i,:]
                                         )
        ACTION["0"] = rpm

        #### Log the simulation ####################################
        LOGGER.log(drone=0, timestamp=i/ENV.SIM_FREQ, state=STATE)

        #### Printout ##############################################
        if i%ENV.SIM_FREQ == 0:
            ENV.render()

        #### Sync the simulation ###################################
        if GUI:
            sync(i, START, ENV.TIMESTEP)

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    LOGGER.save()

    #### Plot the simulation results ###########################
    LOGGER.plot()
