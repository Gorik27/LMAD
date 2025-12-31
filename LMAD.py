from ovito.io import import_file, export_file
from ovito.pipeline import Pipeline, StaticSource
import re
import numpy as np
from subprocess import Popen, PIPE
import logging
from glob import glob
import time
from ase.io import read, write
import os
from pathlib import Path
import shutil
import argparse
from numba import njit
from utils import numerical_sort_key

def find_targets(project, structure, target):
    structure = f'{project}/{structure}'
    atoms = read(structure, format='lammps-data', sort_by_id=False, units="metal")
    inds = np.arange(len(atoms))
    types = np.array(atoms.get_chemical_symbols())
    ind_targets = np.where(types==target)[0]
    return ind_targets


def select_nn(
    project,
    ind0,
    structure,
    outname_selection,
    outname_xyz,
    outname_active,
    cutoff,
    active_radius,
    potential_cutoff):
    structure = f'{project}/{structure}'
    Path(f'{project}/{ind0}').mkdir(parents=True, exist_ok=True)
    outname_selection = f'{project}/{ind0}/{outname_selection}'
    outname_xyz = f'{project}/{ind0}/{outname_xyz}'
    outname_active = f'{project}/{ind0}/{outname_active}'

    """
    Find hopping atom
    """
    atoms = read(structure, format='lammps-data', sort_by_id=False, units="metal")
    inds = np.arange(len(atoms))
    L = np.diag(atoms.cell)
    ids = atoms.arrays['id']
    r0 = atoms.positions[ind0]
    id0 = ids[ind0]
    
    """
    Find neighbors
    """
    ds = atoms.get_distances(ind0, inds, mic=True)

    assert cutoff < (active_radius+potential_cutoff)
    lm_mask = (ds<=cutoff)
    lm_ids = ids[lm_mask]
    N_LMatoms = len(lm_ids)
    print('LM atoms:', N_LMatoms)
    out = 'group lm_atoms id '+' '.join(map(str, lm_ids))
    with open(outname_selection, 'w') as f:
        f.write(out)

    delete_mask = (ds>(active_radius+potential_cutoff))
    del atoms[inds[delete_mask]]
    N_active = len(atoms)
    print('Active atoms:', N_active)

    ids = atoms.arrays['id']
    ind0 = np.where(ids==id0)[0][0] # reassign ind0 in active atoms because we delete some atoms

    types = np.array(atoms.get_chemical_symbols())
    elements = set(atoms.get_chemical_symbols())
    types_num = atoms.arrays['type']
    symbols = ['']*len(elements)
    for e in elements:
        ind = np.where(types==e)[0][0]
        i = types_num[ind]-1
        symbols[i] = types[ind] 

    elements = " ".join(symbols)

    write(outname_active, atoms, format='lammps-data', velocities=True, masses=True, units="metal", specorder=symbols)

    pipeline = import_file(structure)
    data = pipeline.compute()
    data.particles_.create_property('Selection', data=lm_mask)
    pipeline_out = Pipeline(source = StaticSource(data=data))
    export_file(pipeline_out, outname_xyz, "xyz", columns=['Position', 'Particle Type', 'Particle Identifier', 'Selection']) 
    
    return ind0, r0, elements, N_LMatoms

def check_transition(
    project,
    datfiles,
    ind0,
    cutoff,
    transition_all,
    rnd_seed,
    clean_space):
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
            ds = dist_point(r0[ind0], rs[i][ind0], L)
        if np.any(ds>cutoff):
            transition_inds.append(i+1)
            transition_flag = True
            r0 = rs[i]

    if transition_flag:
        print(f'Find transitions: {transition_inds}')
        neb_path = Path(f'{project}/neb/{ind0}/{str(int(rnd_seed))}')
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
                path = f'{project}/qm/{ind0}/{str(int(rnd_seed))}'
                shutil.rmtree(path)
            except OSError as e:
                print(f"Error removing: {path} : {e.strerror}")
            #remove dump files
            try:
                path = f'{project}/dumps/{ind0}/lmad_{str(int(rnd_seed))}'
                shutil.rmtree(path)
            except OSError as e:
                print(f"Error removing: {path} : {e.strerror}")

def md(
    rnd_seed,
    project,
    structure,
    lm_cutoff,
    lmad_steps,
    dump_steps,
    heat_coef,
    elements,
    r0,
    active_radius,
    time_step
):
    routine = 'in.lmad'
    task = (f'{lmp} -in  {routine} \
    -var rnd_seed {rnd_seed} \
    -var target_id {ind0} \
    -var dt_inp {time_step} \
    -var project {project} \
    -var structure {ind0}/{structure} \
    -var lm_radius {lm_cutoff} \
    -var lmad_steps {lmad_steps} \
    -var thermo_step {dump_steps} \
    -var heat_coef {heat_coef} \
    -var elements {elements} \
    -var x0 {r0[0]} \
    -var y0 {r0[1]} \
    -var z0 {r0[2]} \
    -var active_radius {active_radius}')

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

    return dumpfile

def qm(
    project, 
    structure,
    lmp,
    rnd_seed,
    id,
    elements,
    r0,
    active_radius,
    qm_path='qm'
    ):
    routine = 'in.quick_minimize'
    structure = structure.replace(project+'/', '')
    task = (f'{lmp} -in  {routine} -var project {project} \
    -var structure {structure} \
    -var rnd_seed {str(int(rnd_seed))} \
    -var id {id} \
    -var target_id {ind0} \
    -var elements {elements} \
    -var x0 {r0[0]} \
    -var y0 {r0[1]} \
    -var z0 {r0[2]} \
    -var active_radius {active_radius}\
    -var dat_path {qm_path}')
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

    return datfile

def LMAD(
    lmp,
    project,
    structure,
    rnd_seed, 
    heat_coef,
    lmad_steps,
    chech_steps,
    dump_steps,
    cutoff,
    active_radius,
    elements,
    clean_space,
    n_cpu,
    r0, 
    transition_all,
    ind0=None,
    time_step=0.001):

    chech_each = int(chech_steps//dump_steps)
    print(f'random seed: {rnd_seed}')
    """
    STEP 1: LOCAL MELTING
    """
    #print('#1 LM')
    dumpfile = md(rnd_seed,
                project,
                structure,
                lm_cutoff,
                lmad_steps,
                dump_steps,
                heat_coef,
                elements,
                r0,
                active_radius,
                time_step)

    """
    STEP 2: SEARCH FOR TRANSITIONS
    2A: MINIMIZATION
    """
    #print('#2 Unfold')
    #task = f'atomsk --unfold {project}/{dumpfile} -remove-atoms 0 lmp -overwrite'
    type_args = ''
    el_list = [e for e in elements.split(' ') if e!='']
    for i, e in enumerate(el_list):
        type_args += f'-substitute {i+1} {e} '
    task = f'atomsk --unfold {project}/{dumpfile} {type_args} lmp -overwrite'
    with Popen(task.split(), stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            if 'ERROR' in line:
                print(line)

    files = sorted(glob(f'{project}/{dumpfile}*.lmp'), key=numerical_sort_key)
    structures = files[::chech_each]
    #print(structures)

    #print('#3 QM')
    datfiles = []
    for id, structure in enumerate(structures):
        datfile = qm(project, 
                    structure,
                    lmp,
                    rnd_seed,
                    id,
                    elements,
                    r0,
                    active_radius)
        datfiles.append(datfile)
    """
    STEP 2: SEARCH FOR TRANSITIONS
    2B: COMPARIZON
    """           
    #print('#4 Check transitions')
    check_transition(project,
                    datfiles,
                    ind0,
                    cutoff,
                    transition_all,
                    rnd_seed,
                    clean_space)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, 
                        help='name of folder with project')
    parser.add_argument("-s", "--structure", default='relaxed.dat', 
                        help='name of file with prepared structure')
    parser.add_argument("-t", "--target", required=True, 
                        help='target specie [e.g. Ni]; assumed that structure has a single atom of target type which diffusion will be explored')
    parser.add_argument("-lm", "--lm_cutoff", type=float, required=True, 
                        help='atoms with this cutoff [Angstroms] from the target will be heated')
    parser.add_argument("-np", "--np", type=int, default=1)
    parser.add_argument("--lmp", default='lmp_ompi', 
                        help='lammps executable')
    parser.add_argument("-N", '--N_steps', type=int, default=100,
                        help='number of LMAD steps')
    parser.add_argument('--rnd', type=int, nargs='+', default=[],
                        help='random seeds to calculate')
    parser.add_argument('--lmad_steps', type=int, default=500,
                        help='number of MD annealing steps in LMAD run')
    parser.add_argument('--check_steps', type=int, default=100,
                        help='steps interval for checking for transition')
    parser.add_argument('--dump_steps', type=int, default=100,
                        help='steps interval for dumping')
    parser.add_argument('--transition_all', action='store_true', default=False,
                        help='transition event triggered by any atom (True) or only by target atom (False)')
    parser.add_argument("-c", '--distance_cutoff', type=float, default=0.3,
                        help='distance cutoff above which transition is detected [Angstroms]')
    parser.add_argument("-r", '--active_radius', type=float, default=20,
                        help='radius of sphere for MD simulation [Angstroms]')
    parser.add_argument('--potential_cutoff', type=float, default=6,
                        help='radius of sphere for MD simulation [Angstroms]')
    parser.add_argument('--not_clean_space', action='store_false', default=True,
                        help='turn off removing dumps if transition did not occur')
    parser.add_argument("-hc", '--heat_coef', type=float, default=5.0,
                        help='heating coefficient (temperature = melting_temperature*melting_coef)')
    parser.add_argument("-dt", '--time_step', type=float, default=0.001,
                        help='timestep in ps [default: 0.001ps = 1fs]')
    args = parser.parse_args()
    
    project = args.name #'s3_210_1ni'
    target = args.target #'Ni'
    structure = args.structure #'relaxed.dat'
    outname_xyz = 'selection.xyz'
    outname_selection = 'lm_atoms.txt'
    outname_active = 'active.dat'
    lm_cutoff = args.lm_cutoff #4
    active_radius = args.active_radius # 20
    potential_cutoff = args.potential_cutoff

    ind_targets = find_targets(project, structure, target)
    N_targets = len(ind_targets)

    for i, ind in enumerate(ind_targets): 
        print(f'TARGET: {i+1}/{N_targets}')
        ind0, r0, elements, N_LMatoms = select_nn(
            project,
            ind,
            structure,
            outname_selection,
            outname_xyz,
            outname_active,
            lm_cutoff,
            active_radius,
            potential_cutoff*1.1
        )

        print('ind0: ', ind0)
        print('r0: ', r0)
        np.savetxt(f'{project}/{ind0}/select.txt', [ind0, r0[0], r0[1], r0[2]])
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
        clean_space = args.not_clean_space #1#True
        heat_coef = args.heat_coef #5
        time_step = args.time_step #0.001

        with open(f'{project}/params.txt') as f:
            params = {'T': -1, 'T_melt': -1}
            for line in f.readlines():
                match = re.search(r"variable\s+(?P<name>\w+)\s+equal\s+(?P<value>\d*\.?\d+)", line)
                if match:
                    params[match.group("name")] = float(match.group("value"))
        print("parameters:", params)
        if -1 in list(params.values()):
            raise ValueError("params.txt incorrect!")
        kB = 8.617e-5 # eV
        T_lm = params["T_melt"]*heat_coef
        delta_E = 3*kB*(T_lm-params["T"])*N_LMatoms/2
        print(f'LM Temperature: {round(T_lm, 0)}K')
        print(f'Energy transferred to the system: {round(delta_E,1)} eV')

        if len(args.rnd)==0:
            rng_list = range(1, N_steps+1)
        else:
            rng_list = args.rnd
            N_steps = len(rng_list)
        for j, rnd_seed in enumerate(rng_list):
            print(f'STEP: {j+1}/{N_steps}')
            LMAD(
                lmp,
                project,
                outname_active,
                rnd_seed, 
                heat_coef,
                lmad_steps,
                check_steps,
                dump_steps,
                distance_cutoff,
                active_radius,
                elements,
                clean_space,
                n_cpu,
                r0,
                transition_all,
                ind0,
                time_step
            )
