"""Generate reference data, fit local kernel models and evaluate independently."""
import argparse
import json
from pathlib import Path
import time
import CoolProp
import CoolProp.CoolProp as CP
import joblib
import numpy as np
import scipy
import sklearn
from realgas_ml.regressor import LocalRBFRegressor

ROOT = Path(__file__).parent
OUT = ROOT / 'realgas_ml' / 'data'
SIZES = (32, 64, 128, 256)


def grid(size):
    a,b=np.meshgrid(np.linspace(0,1,size),np.linspace(0,1,size),indexing='ij')
    return np.column_stack([a.ravel(),b.ravel()])


def reference(gas, info, X):
    lo,hi=info['T_range_K']; plo,phi=info['P_range_Pa']
    state=CoolProp.AbstractState('HEOS',gas)
    Y=np.empty((len(X),3))
    for i,(t,p) in enumerate(X):
        T=lo+t*(hi-lo); P=plo*(phi/plo)**p
        state.update(CP.PT_INPUTS,P,T)
        if state.phase() not in {CP.iphase_gas,CP.iphase_supercritical_gas,CP.iphase_supercritical}:
            raise ValueError(f'Non-gas state: {gas}, {T}, {P}')
        Y[i]=[np.log(state.compressibility_factor()),state.hmolar(),state.smolar()]
    return Y


def errors(pred, truth, volume_ratio=1.):
    e=np.abs(np.exp(pred[:,0]-truth[:,0])*volume_ratio-1)*100
    return dict(v_MAPE_percent=float(e.mean()),v_max_APE_percent=float(e.max()),
        v_p99_APE_percent=float(np.percentile(e,99)),
        h_MAE_J_mol=float(np.abs(pred[:,1]-truth[:,1]).mean()),
        h_max_AE_J_mol=float(np.abs(pred[:,1]-truth[:,1]).max()),
        s_MAE_J_mol_K=float(np.abs(pred[:,2]-truth[:,2]).mean()),
        s_max_AE_J_mol_K=float(np.abs(pred[:,2]-truth[:,2]).max()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gas', help='Run only one gas for diagnostics')
    args=parser.parse_args()
    meta=json.loads((ROOT/'baseline/metadata.json').read_text())
    meta.update(package_version='0.2.0',method='local quintic RBF kernel regression',
        coolprop_version=CoolProp.__version__,sklearn_version=sklearn.__version__,
        scipy_version=scipy.__version__,seed=20261016,
        data_kind='reference EOS generated; no experimental measurements',
        training_points_per_gas=65536,validation_points_per_gas=512,
        interior_test_points_per_gas=2048,boundary_test_points_per_gas=256,
        selection='Fixed kernel/neighbors; learning curve on validation only. Final test never selects model.',
        model_selection_note='Largest candidate retained to maximize data; validation curve reports whether smaller sizes suffice.')
    metrics={};curve={}
    for idx,(gas,info) in enumerate(meta['fluids'].items()):
        if args.gas and gas != args.gas: continue
        start=time.perf_counter()
        CP.set_reference_state(gas,'DEF')
        info['gas_constant_J_mol_K']=CoolProp.AbstractState('HEOS',gas).gas_constant()
        rng=np.random.default_rng(20261016+idx)
        X=grid(256)
        dev=rng.uniform(0,1,(512,2)); test=rng.uniform(0,1,(2048,2))
        boundary=np.vstack([np.column_stack([np.full(64,side),rng.uniform(0,1,64)]) for side in [0,1]]+
                           [np.column_stack([rng.uniform(0,1,64),np.full(64,side)]) for side in [0,1]])
        y=reference(gas,info,X); yd=reference(gas,info,dev)
        yt=reference(gas,info,test); yb=reference(gas,info,boundary)
        if not np.all(np.isfinite(y)): raise ValueError('Nonfinite labels')
        curve[gas]=[]
        for size in SIZES:
            if size==256: xi,yi=X,y
            else:
                xi=grid(size); yi=reference(gas,info,xi)
            model=LocalRBFRegressor().fit(xi,yi)
            curve[gas].append(dict(train_points=len(xi),**errors(model.predict(dev),yd)))
        baseline=joblib.load(ROOT/'baseline'/f'{gas}.joblib')
        base_pred=np.column_stack([m.predict(test) for m in baseline])
        pred=model.predict(test)
        metrics[gas]=dict(train_points=len(X),validation_points=len(dev),test_points=len(test),
            boundary_test_points=len(boundary),baseline=errors(base_pred,yt,8.31446261815324/info['gas_constant_J_mol_K']),
            improved=errors(pred,yt),boundary=errors(model.predict(boundary),yb),
            elapsed_seconds=time.perf_counter()-start)
        joblib.dump(model,OUT/f'{gas}.joblib',compress=3)
        np.savez_compressed(OUT/f'{gas}_dataset.npz',train_X=X,train_y=y,
            validation_X=dev,validation_y=yd,test_X=test,test_y=yt,boundary_X=boundary,boundary_y=yb)
        print(gas,json.dumps(metrics[gas]),flush=True)
        (OUT/'validation.json').write_text(json.dumps(metrics,indent=2))
        (OUT/'learning_curve.json').write_text(json.dumps(curve,indent=2))
    if not args.gas:
        (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
        (OUT/'reference_data.csv').unlink(missing_ok=True)


if __name__=='__main__': main()
