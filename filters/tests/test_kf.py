import numpy as np
from filters.kf import KF
def test_linear_kf_tracks_constant():
    # 1D: x_dot=0, measure x. Filter should converge to the true constant.
    F=[[0.]];H=[[1.]];Q=[[1e-8]];R=[[0.1]]
    kf=KF(F,H,Q,R,x0=[0.],P0=[[1.]]); rng=np.random.default_rng(0); truth=5.0
    for _ in range(200):
        kf.predict(dt=0.05); kf.update([truth+rng.normal(0,0.3)])
    assert abs(kf.x[0]-truth)<0.15
def test_kf_covariance_shrinks():
    F=[[0.]];H=[[1.]];Q=[[1e-8]];R=[[0.1]]
    kf=KF(F,H,Q,R,x0=[0.],P0=[[1.]]); p0=kf.P[0,0]
    for _ in range(50): kf.predict(dt=0.05); kf.update([5.0])
    assert kf.P[0,0]<p0
