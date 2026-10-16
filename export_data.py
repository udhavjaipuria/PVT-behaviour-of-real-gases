"""Export NPZ datasets to SI-unit CSV. No CoolProp required."""
import argparse
import csv
import gzip
from importlib.resources import files
import numpy as np
from realgas_ml import GasModel, gases
from realgas_ml.model import R

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gas',choices=gases())
    p.add_argument('--output',default='reference_data.csv.gz')
    args=p.parse_args()
    opener=gzip.open if args.output.endswith('.gz') else open
    with opener(args.output,'wt',newline='') as f:
        w=csv.writer(f)
        w.writerow(['gas','split','T_K','P_Pa','Z','v_m3_mol','h_J_mol','s_J_mol_K','source'])
        for gas in [args.gas] if args.gas else gases():
            m=GasModel(gas)
            lo,hi=m.info['T_range_K'];plo,phi=m.info['P_range_Pa']
            with np.load(files('realgas_ml').joinpath(f'data/{gas}_dataset.npz')) as d:
                for split in ['train','validation','test','boundary']:
                    for x,y in zip(d[f'{split}_X'],d[f'{split}_y']):
                        T=lo+x[0]*(hi-lo);P=plo*(phi/plo)**x[1];Z=np.exp(y[0])
                        w.writerow([gas,split,T,P,Z,Z*m.info['gas_constant_J_mol_K']*T/P,y[1],y[2],'CoolProp HEOS 8.0.0 reference EOS'])

if __name__=='__main__':main()
