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
import argparse

def numerical_sort_key(filename):
    # Extract numerical parts from the filename for sorting
    parts = re.findall(r'\d+|\D+', filename)
    return [int(part) if part.isdigit() else part for part in parts]

def select_nn(
    project,
    target,
    structure,
    outname_selection,
    outname_xyz,
    cutoff):
    """
    Find elements
    """
    elements = {}
    num_types = 0
    cnt = 0
    structure = f'{project}/{structure}'
    outname_selection = f'{project}/{outname_selection}'
    outname_xyz = f'{project}/{outname_xyz}'

    with open(structure) as f:
        for line in f.readlines():
            match = re.search(r'\s*(\d+)\s+atom types\s*', line)
            if match:
                num_types = int(match.group(1))
            if num_types:
                if cnt<num_types:
                    match = re.search(r'\s*(\d+)\s+\d+\.\d*\s+#\s+(.+)\s*', line)
                    if match:
                        tokens = match.groups()[::-1]
                        elements.update({tokens[0]: int(tokens[1])})
                        cnt += 1
                elif cnt==num_types:
                    break

    print('find elements:', elements)

    """
    Find hopping atom
    """
    atoms = read(structure, format='lammps-data')
    pts = atoms.positions
    ids = atoms.arrays['id']
    L = np.diag(atoms.cell)
    mask = (atoms.arrays['type']==elements[target])
    id0 = ids[mask][0]
    r0 = pts[mask][0]

    """
    Find neighbors
    """
    #@njit(cache=True)
    def dist_to(rs, pos, L):
        ds = np.zeros(len(rs))
        dr = np.abs(rs-pos)
        for xi in range(3):
            x = dr[:, xi]
            mask = (x>L[xi]*0.5)
            x[mask] = L[xi]-x[mask]
        return np.sum(dr**2, axis=1)**0.5

    ds = dist_to(pts, r0, L)

    lm_mask = (ds<cutoff)
    lm_ids = ids[lm_mask]
    print('Neighboring atoms:', len(lm_ids))

    out = 'group lm_atoms id '+' '.join(map(str, lm_ids))
    with open(outname_selection, 'w') as f:
        f.write(out)

    pipeline = import_file(structure)
    data = pipeline.compute()
    data.particles_.create_property('Selection', data=lm_mask)
    pipeline_out = Pipeline(source = StaticSource(data=data))
    export_file(pipeline_out, outname_xyz, "xyz", columns=['Position', 'Particle Type', 'Particle Identifier', 'Selection']) 
    
    return id0, lm_ids 


def LMAD(
    lmp,
    project,
    rnd_seed, 
    heat_coef,
    lmad_steps,
    chech_steps,
    dump_steps,
    cutoff,
    elements,
    clean_space,
    n_cpu,
    transition_all,
    id0=None):

    chech_each = int(lmad_steps//chech_steps)
    print(f'random seed: {rnd_seed}')
    """
    STEP 1: LOCAL MELTING
    """
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


    def dist_point(r0, r, L):
        dr = np.abs(r-r0)
        mask = (dr>L*0.5)
        dr[mask] = L[mask]-dr[mask]
        ds = np.sqrt(np.sum(dr**2))
        return ds 

    def dist_arrays(r1, r2, L):
        dr = np.zeros_like(r1)
        for i in range(3):
            dr[:, i] = np.abs(r1[:, i]-r2[:, i])
            mask = (dr[:, i]>L[i]*0.5)
            dr[mask, i] = L[i]-dr[mask, i]
        ds = np.sqrt(np.sum(dr**2, axis=1))
        return ds

    transition_inds = [0]
    transition_flag = False
    for i in range(len(rs)):
        if transition_all:
            ds = dist_arrays(r0, rs[i], L)
        else:
            ds = dist_point(r0[id0], rs[i][id0], L)
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
            #remove quick_minimization files
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, 
                        help='name of folder with project')
    parser.add_argument("-s", "--structure", default='relaxed.dat', 
                        help='name of file with prepared structure')
    parser.add_argument("-t", "--target", required=True, 
                        help='target specie [e.g. Ni]; assumed that structure has a single atom of target type which diffusion will be explored')
    parser.add_argument("-o", "--outname_selection", default='lm_atoms.txt', 
                        help='name of file to write lammps group with selected atoms')
    parser.add_argument("-oxyz", "--outname_xyz", default='selection.xyz', 
                        help='name of file to write dump with selected atoms (in exyz format)')
    parser.add_argument("-lm", "--lm_cutoff", type=float, required=True, 
                        help='atoms with this cutoff [Angstroms] from the target will be heated')
    parser.add_argument("-np", "--np", type=int, default=1)
    parser.add_argument("--lmp", default='lmp_ompi', 
                        help='lammps executable')
    parser.add_argument("-N", '--N_steps', type=int, default=100,
                        help='number of LMAD steps')
    parser.add_argument('--lmad_steps', type=int, default=500,
                        help='number of MD annealing steps in LMAD run')
    parser.add_argument('--check_steps', type=int, default=100,
                        help='steps interval for checking for transition')
    parser.add_argument('--dump_steps', type=int, default=10,
                        help='steps interval for dumping')
    parser.add_argument('--transition_all', action='store_true', default=False,
                        help='transition event triggered by any atom (True) or only by target atom (False)')
    parser.add_argument("-c", '--distance_cutoff', type=float, default=0.3,
                        help='distance cutoff above which transition is detected [Angstroms]')
    parser.add_argument('--not_clean_space', action='store_false', default=True,
                        help='turn off removing dumps if transition did not occur')
    parser.add_argument("-hc", '--heat_coef', type=float, default=5.0,
                        help='heating coefficient (temperature = melting_temperature*melting_coef)')
    parser.add_argument('-e', '--elements', type=str, nargs='+', required=True,
                        help='chemical elements separated by spaces (to put in lammps potential setup)')
    args = parser.parse_args()
    
    project = args.name #'s3_210_1ni'
    target = args.target #'Ni'
    structure = args.structure #'relaxed.dat'
    outname_xyz = args.outname_xyz #'selection.xyz'
    outname_selection = args.outname_selection #'lm_atoms.txt'
    lm_cutoff = args.lm_cutoff #4

    id0, lm_ids = select_nn(
        project,
        target,
        structure,
        outname_selection,
        outname_xyz,
        lm_cutoff
    )

    n_cpu = args.np #8S
    lmp_bin = args.lmp #'lmp_ompi'
    if n_cpu > 1:
        lmp = f'mpiexec --np {n_cpu} {lmp_bin}'
    elif n_cpu == 1:
        lmp = lmp_bin
    else:
        raise ValueError(f'number of cpus must be positive integer, not {n_cpu}!!!')

    N_steps = args.N_steps #100

    lmad_steps = args.lmad_steps #500
    check_steps = args.check_steps #100
    dump_steps = args.dump_steps #10
    transition_all = args.transition_all #False # True - transition event triggered by any atom, False - only by target atom (id0)

    distance_cutoff = args.distance_cutoff #0.3 #A
    clean_space = (not args.not_clean_space) #1#True
    heat_coef = args.heat_coef #5
    elements = ' '.join(args.elements) #"Ag Ni"

    rng_list = range(1, N_steps+1)
    for rnd_seed in rng_list:
        print(f'STEP: {rnd_seed}/{N_steps}')
        LMAD(
            lmp,
            project,
            rnd_seed, 
            heat_coef,
            lmad_steps,
            check_steps,
            dump_steps,
            distance_cutoff,
            elements,
            clean_space,
            n_cpu,
            transition_all,
            id0
        )
