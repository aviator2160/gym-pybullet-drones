# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:14:06 2021

@author: Sarah Li
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

_STATES_DICT = {
    'north': 0,
    'east':  1,
    'down':  2,
    'pitch': 3,
    'roll':  4,
    'yaw':   5, 
    'north velocity': 6, 
    'east velocity': 7, 
    'down velocity': 8,
    'pitch velocity': 9,
    'roll velocity': 10,
    'yaw velocity': 11
}
    
def slung_dynamics_gen(mass, J, g_list):
    """ Generate the multi-lift slung load hovering dynamics.
    
    A payload is carried by multiple smaller vehicles. Dynamics are described
    in North-East-Down frame with Euler angles. See Chapter 2 of 
    https://etda.libraries.psu.edu/files/final_submissions/10269
    for further description.
        dot_x = Ax + Bu
    Dynamic variables (x):
        - (n, e, d): paylod's Center of gravity (CoG) in the NED frame.
        - (phi, theta, psi): Pitch, roll, yaw Euler angles.
        - (v_n, v_e, v_d): Velocities of payload's CoG in NED axis directions.
        - (p, q, r): Angular velocity of payload's Euler angles. 
    
    Control is defined for each smaller carrier vehicle i. 
    Control variables (u_i):
        - (Fn, Fe, Fd): Forces in NED framework exerted by vehicle i. 
        
    Args:
        - mass: Mass of the payload.
        - J: 3D ndarray vector of payload's inertia at its center of gravity.
        - g_list: list of the locations of payload's cable connections to
        lifting UAVs, centered at the payload's CoG.
    Returns:
        - A: Continuous time system dynamics at hovering. 
        - [B_1,...B_m]: Control matrices for each carrier vehicle. 
    """
    n = 12 # number of independent dynamic variables.
    m = 6 # number of independent input variables per carrier vehicle.
    A = np.zeros((n, n))
    B = np.zeros((n,m))
    A[0:3, 6:9] = np.eye(3)
    A[3:6, 9:12] = np.eye(3)
    B[6:9, 0:3] = 1/mass*np.eye(3)
    B[9:12, 3:6] = la.inv(J)
    B_list = []
    for g in g_list:
        G_i = np.zeros((m, 3))
        G_i[0:3, 0:3] = np.eye(3)
        G_i[3,1] = -g[2] # -g_z
        G_i[3,2] = g[1] # g_y
        G_i[4,0] = g[2] # g_z
        G_i[4,2] = -g[0] # -g_x
        G_i[5,0] = -g[1] # -g_y
        G_i[5,1] = g[0] # g_x
        B_list.append(B.dot(G_i))
        
    return A, B_list

def state_matrix(states_list):
    """ Return a matrix for the slung load dynamics. 
    Args:
        - states_list: a list of states that should be in the range [str list].
    Returns:
        - H: the corresponding matrix [ndarray, nxl].
    """
    n = len(_STATES_DICT)
    l = len(states_list)
    H = np.zeros((n, l))
    for ind in range(l):
        state_ind = _STATES_DICT[states_list[ind]]
        H[state_ind, ind] = 1.
    return H