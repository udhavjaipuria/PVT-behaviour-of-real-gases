from importlib.resources import files
import numpy as np
import pytest
from realgas_ml import GasModel, gases

@pytest.mark.parametrize('gas', gases())
def test_held_out_reference_accuracy(gas):
    m=GasModel(gas)
    with np.load(files('realgas_ml').joinpath(f'data/{gas}_dataset.npz')) as d:
        assert len(d['train_X'])==65536
        assert len(d['validation_X'])==512
        assert len(d['test_X'])==2048
        for split in ['test','boundary']:
            pred=m.model.predict(d[f'{split}_X']);truth=d[f'{split}_y']
            error=np.abs(np.expm1(pred[:,0]-truth[:,0]))*100
            assert error.max()<0.001
            assert error.mean()<0.0001
            assert np.abs(pred[:,1]-truth[:,1]).mean()<0.01
            assert np.abs(pred[:,2]-truth[:,2]).mean()<0.0001


def test_scalar_api_matches_dataset():
    m=GasModel('CarbonDioxide')
    with np.load(files('realgas_ml').joinpath('data/CarbonDioxide_dataset.npz')) as d:
        for x,truth in zip(d['test_X'][:10],d['test_y'][:10]):
            lo,hi=m.info['T_range_K'];plo,phi=m.info['P_range_Pa']
            T=lo+x[0]*(hi-lo);P=plo*(phi/plo)**x[1]
            r=m.predict(T,P)
            assert r['Z']==pytest.approx(np.exp(truth[0]),rel=1e-6)
            assert r['h_J_mol']==pytest.approx(truth[1],abs=0.01)


def test_train_test_disjoint():
    for gas in gases():
        with np.load(files('realgas_ml').joinpath(f'data/{gas}_dataset.npz')) as d:
            train=set(map(tuple,d['train_X']))
            unseen=[]
            for split in ['validation','test','boundary']:
                points=set(map(tuple,d[f'{split}_X']))
                assert not train.intersection(points)
                assert all(not points.intersection(p) for p in unseen)
                unseen.append(points)


@pytest.mark.parametrize('gas', gases())
def test_volume_against_source_eos(gas):
    CP=pytest.importorskip('CoolProp.CoolProp')
    m=GasModel(gas)
    lo,hi=m.info['T_range_K']
    for T,P in [(lo,1e5),((lo+hi)/2,2e6),(hi,5e6)]:
        r=m.predict(T,P)
        v=1/CP.PropsSI('Dmolar','T',T,'P',P,gas)
        assert r['v_m3_mol']==pytest.approx(v,rel=1e-6)


@pytest.mark.parametrize('gas', gases())
def test_positive_compressibility_on_check_grid(gas):
    m=GasModel(gas)
    # A finite grid check, not a global thermodynamic-stability guarantee.
    t,p=np.meshgrid(np.linspace(0,1,16),np.linspace(0,1,128),indexing='ij')
    X=np.column_stack([t.ravel(),p.ravel()])
    y=m.model.predict(X)
    volume_factor=np.exp(y[:,0]-X[:,1]*np.log(50)).reshape(16,128)
    assert np.all(np.diff(volume_factor,axis=1)<0)
