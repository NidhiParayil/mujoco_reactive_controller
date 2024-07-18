#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Solve model predictive control problems."""

from qpsolvers import solve_problem

from mpc_problem import MPCProblem
from mpc_qp import MPCQP
from plan import Plan
import numpy as np
# import pylab
import matplotlib.pyplot as plt


def solve_mpc(
    problem: MPCProblem,
    solver: str,
    sparse: bool = False,
    **kwargs,
) -> Plan:
    """Solve a linear time-invariant model predictive control problem.

    Args:
        problem: Model predictive control problem to solve.
        solver: Quadratic programming solver to use, to choose in
            :data:`qpsolvers.available_solvers`. Empirically the best
            performing solvers are Clarabel and ProxQP: see for instance this
            `benchmark of QP solvers for model predictive control
            <https://github.com/qpsolvers/mpc_qpbenchmark>`__.
        sparse: Whether to use sparse or dense matrices in the output quadratic
            program. Enable it if the QP solver is sparse (e.g. OSQP).
        kwargs: Keyword arguments forwarded to the QP solver via the
            `solve_qp`_ function.

    Returns:
        Solution to the problem, if found.

    .. _solve_qp:
        https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
    """
    mpc_qp = MPCQP(problem, sparse=sparse)
    qpsol = solve_problem(mpc_qp.problem, solver=solver, **kwargs)
    return Plan(problem, qpsol)

horizon_duration = 1.0
nb_timesteps = 16
T = horizon_duration / nb_timesteps
A = np.array([[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]])
B = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T]).reshape((3, 1))

# Acceleration limits
accel_from_state = np.array([0.0, 0.0, 1.0])
max_accel = 3.0  # [m] / [s] / [s]
ineq_matrix = np.vstack([+accel_from_state, -accel_from_state])
ineq_vector = np.array([+max_accel, +max_accel])

# Complete LTV problem
initial_pos = 0.0
goal_pos = 1.0
problem = MPCProblem(
    transition_state_matrix=A,
    transition_input_matrix=B,
    ineq_state_matrix=ineq_matrix,
    ineq_input_matrix=None,
    ineq_vector=ineq_vector,
    initial_state=np.array([initial_pos, 0.0, 0.0]),
    goal_state=np.array([goal_pos, 0.0, 0.0]),
    nb_timesteps=nb_timesteps,
    terminal_cost_weight=1.0,
    stage_state_cost_weight=None,
    stage_input_cost_weight=1e-6,
)

# Solve our LTV problem
plan = solve_mpc(problem, solver="osqp")

# Plot solution
fig, axs = plt.subplots(3, 1)
t = np.linspace(0.0, horizon_duration, nb_timesteps + 1)
X = plan.states
positions, velocities, accelerations = X[:, 0], X[:, 1], X[:, 2]

axs[0].plot(t, positions)
axs[1].plot(t, velocities)
axs[2].plot(t, accelerations)

plt.suptitle("position velocity acceleration")
plt.savefig("test.png")
print("all good so far")