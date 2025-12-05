from ovito.io import import_file, export_file
from ovito.pipeline import Pipeline, StaticSource
import re
import numpy as np

project = 's3_210_1ni/'
target = 'Ni'
fname = project+'relaxed.dat'
fname_out = project+'selection.xyz'
outfile = project+'lm_atoms.txt'
cutoff = 4

"""
Find elements
"""
elements = {}
num_types = 0
cnt = 0
with open(fname) as f:
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
Find hoppin atom
"""
pipeline = import_file(fname)
data = pipeline.compute()
prts = data.particles
mask = (prts['Particle Type']==elements[target])
ids = prts['Particle Identifier'].array
pts = prts['Position'].array

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

L = np.diag(data.cell)
ds = dist_to(pts, r0, L)

lm_mask = (ds<cutoff)
ids = prts['Particle Identifier']
lm_ids = ids[lm_mask]
print('Neighboring atoms:', len(lm_ids))

out = 'group lm_atoms id '+' '.join(map(str, lm_ids))
with open(outfile, 'w') as f:
    f.write(out)

data.particles_.create_property('Selection', data=lm_mask)
pipeline_out = Pipeline(source = StaticSource(data=data))
export_file(pipeline_out, fname_out, "xyz", columns=['Position', 'Particle Type', 'Particle Identifier', 'Selection'])