import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpltools import annotation

matplotlib.use("Qt5Agg")

def plot_convergence(m_nonlin, m_lin, plot_errors=False):
    iters = np.arange(len(m_lin._residuals))
    iters_nonl = np.arange(len(m_nonlin._residuals))

    plt.semilogy(iters, m_lin._residuals, label="linear")
    plt.semilogy(iters_nonl, m_nonlin._residuals, label="nonlinear")
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    # Indicate convergence order
    # order = 1
    # annotation.slope_marker((iters[2], 0.8 * m_nonlin._residuals[4]), (-1 - order), ax=ax, invert=True, size_frac=1/nsteps)
    plt.legend()
    if plot_errors:
        plt.figure()
        p_analytical = m_nonlin.p_analytical()
        e_nonlin = [np.linalg.norm(p_num - p_analytical) for p_num in m_nonlin._solutions]
        e_lin = [np.linalg.norm(p_num - p_analytical) for p_num in m_lin._solutions]

        plt.semilogy(iters, e_lin, label="linear")
        plt.semilogy(iters_nonl, e_nonlin, label="nonlinear")
        ax = plt.gca()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error (not normalized)")
        plt.legend()
    plt.show()