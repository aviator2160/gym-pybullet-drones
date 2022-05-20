# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019
@author: Sarah Li
"""
import numpy as np
import cvxpy as cvx
import scipy.linalg as sla

import gym_pybullet_drones.control.subspace as subspace

def isa(V0, A, B, eps = 0.0, verbose=False):
    K = V0
    Vt= 1.0*V0
    isInvariant = False
    iteration = 0
    while not isInvariant:
        iteration += 1
        if verbose:
            print ("--------------in interation ", iteration)

        Ainved =  subspace.a_inv_v(A, subspace.sum(Vt, B))
        Vnext = subspace.intersect(K, Ainved)
        if verbose:
            print ("rank of Vt ", subspace.rank(Vt))
            print ("rank of Vnext ", subspace.rank(Vnext))
            print ("Vnext is contained in Vt ", subspace.contained(Vnext, Vt))
        if subspace.rank(Vnext) == subspace.rank(Vt):
            isInvariant = True
            if verbose:
                print ("ISA returns-----") 
        else:
            Vt = Vnext
        
    return Vt

def disturbance_decoupling(H, A, B, return_alpha=False, verbose=False):
    """ Generate a distrubance decoupling controller F. 
    
    Args:
        - H: the observation matrix [ndarray, lxn]. 
        - A: system dynamics matrix [ndarray, nxn].
        - B: control matrix [ndarray, nxm].
        - return_alpha: True if you want the corresponding laypunov pontential.
        - verbose: True if want to see cvx output, False otherwise.
    Returns:
        - V: a matrix whose range is largest (A,B) control invariant set in
             the kernel of H.T [ndarray, nxk].
        - F: a disturbance decoupling optimal controller [ndarray, mxn].
        - alpha: magnitude of Lyapunov potential if needed [float].
        - P: the lyapunov potential matrix [ndarray nxn]. 
    """        
    V = isa(subspace.ker(H.T), A, B)
    VB = np.concatenate((V,B), axis=1)
    v_row, v_col = V.shape
    n,m = B.shape
    F = cvx.Variable((m, n))
    XS = cvx.Variable((v_col + m, v_col))
    constraint = [VB@XS == A.dot(V)]
    constraint.append(XS[v_col:, :] == -F@V) 
    dd_opt = cvx.Problem(cvx.Minimize(0), constraint)
    dd_opt.solve(solver=cvx.MOSEK, verbose=verbose) 
    
    # abstract resulting controller and confirm constraints.
    F_array = F.value
    XS_array = XS.value
    X_0 = XS_array[:v_col, :]
    lhs = V.dot(X_0) - B.dot(F_array).dot(V)
    rhs = A.dot(V)
    print(f'DD condition is satisfied {np.allclose(lhs,rhs)}')
    alpha = None
    P = None
    if return_alpha: # solve Lyapunove euqation: 
        A_cl = A+ B.dot(F_array)
        e, v = sla.eig(A_cl.T + A_cl)
        alpha = max(np.abs(e)) * -0.5
        VB = np.concatenate((V, B),axis=1)
        _,v_col = V.shape
        P = np.eye(n)
    return V, F_array, alpha, P

def solve_bmi(A, B, alpha_0, F_0, P_0, V = None, rho = 1e-2, verbose=False):
    """ Solve the stabilizing BMI with disturbance decoupling.
    
    Set V = None for standard output feedback control
    
    Args:
        - A: system dynamics [ndarray, nxn].
        - B: controller dynamics [ndarray, nxm].
        - F_0: initial controller, must satisfy disturbance decoupling,
            [ndarray, mxn].
        - alpha_0: initial Lyapunov function bound.
        - P_0: initial Lyapunov function.
        - V: largest invariant set for disturbanc decoupling [ndarray, nxk].
        - rho: penalty factor for optimization during iterative process[float].
        - verbose: True if print output algorithm updates.
    Returns:
        - F_k: the optimal controller that solves the BMI [ndarray, mxn].
        - alpha_k: the optimal alpha value [float].
        - P_k: the Lyapunov potential matrix [ndarray, nxn].
        - eig_history: a list of the alpha value derived over the algorithm.
    """
    n,m  = B.shape
    if V is not None:
       v_row, v_col = V.shape
       VB = np.concatenate((V,B), axis=1)
       XS = cvx.Variable((v_col + m, v_col))
    F = cvx.Variable((m, n))
    P = cvx.Variable((n,n), PSD=True)
    
    alpha = cvx.Variable()
    a_I = alpha * np.eye(n)
    rho = 1e-2
    eig_history = []
    T = 50
    def H(alpha, F, P):
        return  A + alpha * np.eye(n) + B.dot(F) - P
    
    alpha_k = alpha_0
    F_k = F_0
    P_k = P_0
    for k in range(T):
        e, v = sla.eig(A + B.dot(F_k))
        # print(e + alpha.value)
        # if verbose:
            # print(f'\r eigen values = {np.round(e + alpha.value, 2)}', end = '         ')
            # print(f"Eigen-values = {np.max(np.real(e))}")
            # print(f"alpha =  {alpha.value}")
            # print(f"--------- {k} -----------")
        eig_history.append(np.max(np.real(e)))
        H_k = H(alpha_k, F_k, P_k)
        H_var = H_k @ (2 * (A + a_I + B@F - P) - H_k)
        lmi_block = cvx.bmat([[H_var, (A + B@F + a_I  + P).T],
                              [A + B@F + a_I  + P,  np.eye(n)]])
        obj = cvx.Maximize(alpha-rho/2 * cvx.power(cvx.norm(F-F_k, 'fro'),2) 
                           - rho/2 * cvx.power(cvx.norm(P - P_k, 'fro'), 2) )
        constraint = [lmi_block >> 0] # Lyapunov stability constraint.
        if V is not None: # solve with disturbance decoupling
            constraint.append(VB@XS == A.dot(V))
            constraint.append(XS[v_col:, :] == -F@V)
        k_iteration = cvx.Problem(obj, constraint)
        k_iteration.solve(solver=cvx.MOSEK, verbose=False)
        F_k = F.value
        P_k = P.value
        alpha_k = alpha.value
        pe, pv = sla.eig(P_k)
        if verbose:
            print(f'\r eigen values = {np.round(e + alpha.value, 2)}, P eigs = {np.round(pe, 2)}', end = '         ')
    if V is not None and verbose==True:    
        # check disturbance decoupling is satisfied
        XS_array = XS.value
        X_0 = XS_array[:v_col, :]
        lhs = V.dot(X_0) - B.dot(F_k).dot(V)
        rhs = A.dot(V)
        print(f'DD condition is satisfied {np.allclose(lhs,rhs)}')
    
    return F_k, alpha_k, P_k, eig_history