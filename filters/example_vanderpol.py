"""Worked EKF example: Van der Pol oscillator, estimating the UNMEASURED velocity
from noisy position measurements. Demonstrates the core EKF value proposition.
"""
import numpy as np
from filters.ekf import EKF

MU = 1.0
def f(x, u, t): return np.array([x[1], MU*(1-x[0]**2)*x[1] - x[0]])
def F(x, u, t): return np.array([[0.,1.],[-2*MU*x[0]*x[1]-1., MU*(1-x[0]**2)]])
def h(x, t):    return np.array([x[0]])
def H(x, t):    return np.array([[1.,0.]])

def run(seed=0, T=20.0, dt=0.01, meas_every=20):
    rng = np.random.default_rng(seed)
    Q = np.diag([1e-6,1e-4]); R = np.array([[0.04]])
    x_true = np.array([2.0,0.0])
    ekf = EKF(f,F,h,H,Q,R, x0=np.array([0.,0.]), P0=np.diag([2.,2.]))
    N=int(T/dt); H_={k:[] for k in ('t','x1t','x2t','x1h','x2h','trP','mt','my')}
    for k in range(N):
        t=k*dt
        a1=f(x_true,None,t);a2=f(x_true+0.5*dt*a1,None,t);a3=f(x_true+0.5*dt*a2,None,t);a4=f(x_true+dt*a3,None,t)
        x_true=x_true+(dt/6)*(a1+2*a2+2*a3+a4)
        ekf.predict(None,t,dt)
        if k%meas_every==0:
            y=h(x_true,t)+rng.normal(0,np.sqrt(R[0,0]),1); ekf.update(y,t)
            H_['mt'].append(t); H_['my'].append(y[0])
        H_['t'].append(t); H_['x1t'].append(x_true[0]); H_['x2t'].append(x_true[1])
        H_['x1h'].append(ekf.x[0]); H_['x2h'].append(ekf.x[1]); H_['trP'].append(np.trace(ekf.P))
    return {k:np.array(v) for k,v in H_.items()}

if __name__=="__main__":
    r=run(); print("pos err %.4f  VEL err %.4f (never measured)"%(abs(r['x1t']-r['x1h'])[-1], abs(r['x2t']-r['x2h'])[-1]))
