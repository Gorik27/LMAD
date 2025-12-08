from ovito.io import import_file, export_file
from ovito.pipeline import Pipeline, StaticSource
import re
import numpy as np
from subprocess import Popen, PIPE
import logging
from glob import glob
import time
from ase.io import read
import os
from pathlib import Path
import shutil


def numerical_sort_key(filename):
    # Extract numerical parts from the filename for sorting
    parts = re.findall(r'\d+|\D+', filename)
    return [int(part) if part.isdigit() else part for part in parts]


n_cpu = 8
project = 's3_210_1ni'
#rnd_seed = 3
N_steps = 10

lmad_steps = 500
chech_steps = 100
chech_each = int(lmad_steps//chech_steps)
dump_steps = 10

cutoff = 0.3 #A
clean_space = 1#True
heat_coef = 5
elements = "Ag Ni"

lmp = f'mpiexec --np {n_cpu} lmp_ompi'

rng_list = range(1, N_steps+1)
for rnd_seed in rng_list:
    """
    STEP 1: LOCAL MELTING
    """
    print(f'Step {rnd_seed}')
    routine = 'in.lmad'
    task = (f'{lmp} -in  {routine} \
    -var rnd_seed {rnd_seed} \
    -var project {project} \
    -var heat_coef {heat_coef} \
    -var elements {elements} \
    -var thermo_steps {dump_steps} \
    -var lmad_steps {lmad_steps}')

    finished = False
    dumpfile = ''
    with Popen(task.split(), stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        time.sleep(0.1)
        #print('\n')
        #logging.info('\n')
        for line in p.stdout:
            if 'ERROR' in line:
                raise ValueError(f'ERROR in LAMMPS: {line}')
            if 'All done!' in line:
                finished = True
            elif "dumpfile" in line:
                dumpfile = (line.replace('dumpfile: ', '')).replace('\n', '')

    if not finished:
        raise ValueError('ERROR!!!\n Something went wrong during LMAD')

    """
    STEP 2: SEARCH FOR TRANSITIONS
    2A: MINIMIZATION
    """
    #files = sorted(glob(f'{project}/{dumpfile}'), key=numerical_sort_key)
    #structures = files[::chech_each]
    #print(structures)

    #s_name = structure.replace('dump.', '')
    task = f'atomsk --unfold {project}/{dumpfile} lmp -overwrite'
    with Popen(task.split(), stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            if 'ERROR' in line:
                print(line)

    files = sorted(glob(f'{project}/{dumpfile}*.lmp'), key=numerical_sort_key)
    structures = files[::chech_each]
    #print(structures)

    routine = 'in.quick_minimize'
    datfiles = []
    for id, structure in enumerate(structures):
        #print(1+id, '/', len(structures))
        
        structure = structure.replace(project+'/', '')
        task = (f'{lmp} -in  {routine} -var project {project} -var structure {structure} -var rnd_seed {str(int(rnd_seed))} -var id {id} -var elements {elements}')
        with Popen(task.split(), stdout=PIPE, bufsize=1, universal_newlines=True) as p:
            time.sleep(0.1)
            #print('\n')
            #logging.info('\n')
            for line in p.stdout:
                if 'ERROR' in line:
                    raise ValueError(f'ERROR in LAMMPS: {line}')
                if 'All done!' in line:
                    finished = True
                elif "datfile" in line:
                    datfile = (line.replace('datfile: ', '')).replace('\n', '')
        if not finished:
            raise ValueError('ERROR!!!\n Something went wrong during minimization')
        datfiles.append(datfile)
    """
    STEP 2: SEARCH FOR TRANSITIONS
    2B: COMPARIZON
    """                
    #print(datfiles)

    def get_rs(file):
        atoms = read(f'{project}/{file}', format='lammps-data')
        return atoms.positions

    atoms = read(f'{project}/{datfiles[0]}', format='lammps-data')
    r0 = atoms.positions
    L = np.diag(atoms.cell)
    rs = [get_rs(datfile) for datfile in datfiles[1:]]

    """     def dist(r1, r2, L):
        dr = np.zeros_like(r1)
        for i in range(3):
            dr[:, i] = np.abs(r1[:, i]-r2[:, i])
            mask = (dr[:, i]>L[i]*0.5)
            dr[mask, i] = L[i]-dr[mask, i]
        ds = np.sqrt(np.sum(dr**2, axis=1))
        return ds """

    def dist(r0, r, L):
        dr = np.zeros(3)
        for i in range(3):
            dr[i] = np.abs(r[i]-r0[i])
            if (dr[i]>L[i]*0.5):
                dr[i] = L[i]-dr[i]
        ds = np.sqrt(np.sum(dr**2))
        return ds 

    transition_inds = [0]
    transition_flag = False
    for i in range(len(rs)):
        #ds = dist(r0, rs[i], L)
        ds = dist(r0[id0], rs[i][id0], L)
        if np.any(ds>cutoff):
            transition_inds.append(i+1)
            transition_flag = True
            r0 = rs[i]

    if transition_flag:
        print(f'Find transitions: {transition_inds}')
        neb_path = Path(f'{project}/neb/{str(int(rnd_seed))}')
        neb_path.mkdir(parents=True, exist_ok=True)
        for i, ind in enumerate(transition_inds):
            src = f'{project}/{datfiles[ind]}'
            dst = neb_path/f'{i}.dat'
            shutil.copy(src, dst)
    else:
        transition_inds = []
        print('There is no transitions')
        if clean_space:
            #remove qm files
            try:
                path = f'{project}/qm/{str(int(rnd_seed))}'
                shutil.rmtree(path)
            except OSError as e:
                print(f"Error removing: {path} : {e.strerror}")
            #remove dump files
            try:
                path = f'{project}/dumps/lmad_{str(int(rnd_seed))}'
                shutil.rmtree(path)
            except OSError as e:
                print(f"Error removing: {path} : {e.strerror}")





