"""
A small, dependency-free SPSA optimizer (Simultaneous Perturbation Stochastic
Approximation), used to train the PennyLane circuits in qml_core.py the same
way the original notebook trains the Qiskit circuits with
``qiskit_machine_learning.optimizers.SPSA``.

We don't reuse PennyLane's built-in ``qml.SPSAOptimizer`` here because it
expects ``pennylane.numpy`` arrays with ``requires_grad`` set and an
objective function returning an array-like (not a plain float) -- plain
gradient-free SPSA is simple enough to implement directly and keeps the
classical training loop framework-agnostic.
"""
import numpy as np


class OptimizerLog:
    """Records SPSA progress and evaluates train/test accuracy each iteration."""

    def __init__(self, evaluate_fn):
        """
        evaluate_fn(parameters) -> (train_accuracy, test_accuracy)
        """
        self.evaluate_fn = evaluate_fn
        self.evaluations, self.parameters, self.costs = [], [], []
        self.train_accuracies, self.test_accuracies, self.stepsizes = [], [], []

    def update(self, evaluation, parameter, cost, stepsize):
        self.evaluations.append(evaluation)
        self.parameters.append(np.array(parameter, copy=True))
        self.costs.append(float(cost))
        self.stepsizes.append(float(stepsize))
        tr, te = self.evaluate_fn(parameter)
        self.train_accuracies.append(tr)
        self.test_accuracies.append(te)
        print(f" Iter {len(self.evaluations):3d}: loss={cost:.4f}  "
              f"train_acc={tr:.4f}  test_acc={te:.4f}  step={stepsize:.4f}")


def spsa_minimize(cost_fn, initial_point, maxiter, log=None,
                   alpha=0.602, gamma=0.101, c=0.2, a=None, A=None, seed=None):
    """
    Minimize ``cost_fn(params) -> float`` with plain SPSA.

    a_k = a / (A + k + 1)^alpha      (parameter-update step size)
    c_k = c / (k + 1)^gamma          (perturbation size)
    """
    rng = np.random.default_rng(seed)
    theta = np.array(initial_point, dtype=float, copy=True)
    n = len(theta)

    if A is None:
        A = max(1.0, 0.1 * maxiter)
    if a is None:
        # Calibrate `a` from a handful of random-direction gradient estimates so
        # that the first step moves each parameter by roughly `target_step`.
        target_step = 0.2
        c0 = c
        grad_mags = []
        for _ in range(5):
            delta = rng.choice([-1.0, 1.0], size=n)
            yplus = cost_fn(theta + c0 * delta)
            yminus = cost_fn(theta - c0 * delta)
            ghat = (yplus - yminus) / (2 * c0) * delta
            grad_mags.append(np.mean(np.abs(ghat)))
        mean_grad = max(np.mean(grad_mags), 1e-8)
        a = target_step * (A + 1) ** alpha / mean_grad

    final_cost = None
    for k in range(maxiter):
        ak = a / (A + k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=n)

        yplus = cost_fn(theta + ck * delta)
        yminus = cost_fn(theta - ck * delta)
        ghat = (yplus - yminus) / (2 * ck) * delta

        theta = theta - ak * ghat
        final_cost = cost_fn(theta)

        if log is not None:
            log.update(evaluation=k, parameter=theta, cost=final_cost, stepsize=ak)

    return theta, final_cost
