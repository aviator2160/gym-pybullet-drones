import numpy as np
import scipy.linalg as sla
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
import gym_pybullet_drones.control.slung as slung
import gym_pybullet_drones.control.disturbance_decouple as dd

class DDLoadControl():
    """
    
    """

    ################################################################################

    def __init__(self,
                 load_params,
                 g: float=9.8,
                 doNotStabilize: bool=False
                 ):
        """Control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        doNotStabilize
            Stops generation of a stabilizing controller, which is a lengthy process. For debugging.

        """
        #### Set general use constants #############################
        self.MASS = load_params.mass
        self.J = load_params.J
        self.HARDPOINTS = load_params.hardpoints
        """ """
        self.GRAVITY = g*self.MASS
        """float: The gravitational force (M*g) acting on each drone."""
        self.doNotStabilize = doNotStabilize
        self.reset()
        self.F = self.getDDController()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ####

    def getDDController(self):
        # slung dynamics generation.
        A, B_list = slung.slung_dynamics_gen(self.MASS, self.J, self.HARDPOINTS)
        B = np.concatenate(B_list, axis=1)
        H = slung.state_matrix(['down', 'pitch', 'roll'])

        #--------------------- solve disturbance decoupling -----------------------#
        V, F_0, alpha_0, P_0 = dd.disturbance_decoupling(H, A, B,return_alpha=True)
        if self.doNotStabilize:
            return F_0
        #--------------------- solve BMI -----------------------#
        F_k, alpha_k, P_k, eig_history_with_dd = dd.solve_bmi(A, B, alpha_0, F_0, P_0, V,
                                                      verbose=False)

        # final result should give the fastest convergence rate of A+BF that's also DD
        # A_cl = A + B.dot(F_k)
        # e, v = sla.eig(A_cl)
        # print(f"Eigen-values of closed loop system = {e}")
        
        # E = slung.state_matrix(['north velocity', 'east velocity'])

        return F_k
    
    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """ 
        cur_eul = p.getEulerFromQuaternion(cur_quat)
        cur_state = np.concatenate([cur_pos, cur_eul, cur_vel, cur_ang_vel])
        forces = self.F @ cur_state
        return forces