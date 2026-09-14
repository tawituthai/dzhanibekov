import numpy as np
from filters.ekf import EKF
from filters.example_vanderpol import run, f, F, h, H

def test_estimate_converges():
    r=run(seed=0); assert np.mean(np.abs(r['x1t']-r['x1h'])[-500:]) < 0.1
def test_unmeasured_velocity_recovered():
    r=run(seed=0); assert np.mean(np.abs(r['x2t']-r['x2h'])[-500:]) < 0.15
def test_covariance_shrinks():
    r=run(seed=1); assert r['trP'][-1] < r['trP'][0]
def test_covariance_positive_definite():
    rng=np.random.default_rng(3); Q=np.diag([1e-6,1e-4]); R=np.array([[0.04]])
    ekf=EKF(f,F,h,H,Q,R,x0=[0.,0.],P0=np.diag([2.,2.])); x=np.array([2.,0.]); dt=0.01
    for k in range(2000):
        t=k*dt; a1=f(x,None,t);a2=f(x+0.5*dt*a1,None,t);a3=f(x+0.5*dt*a2,None,t);a4=f(x+dt*a3,None,t)
        x=x+(dt/6)*(a1+2*a2+2*a3+a4); ekf.predict(None,t,dt)
        if k%20==0: ekf.update(h(x,t)+rng.normal(0,0.2,1),t)
        assert np.all(np.linalg.eigvalsh(ekf.P)>0)
