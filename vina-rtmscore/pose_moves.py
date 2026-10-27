"""PDBQT branch-tree mutations; no crystal coordinates or bond inference."""
from dataclasses import dataclass
import numpy as np


class InvalidPoseError(ValueError):
    """A geometrically invalid local proposal, not an infrastructure failure."""


def rotate(coords, axis, angle, origin):
    axis=np.asarray(axis,dtype=float)
    norm=np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-12 or not np.isfinite(angle):
        raise ValueError('Invalid rotation')
    axis=axis/norm
    x=np.asarray(coords)-origin
    return origin+x*np.cos(angle)+np.cross(axis,x)*np.sin(angle)+np.outer(x@axis,axis)*(1-np.cos(angle))


@dataclass(frozen=True)
class Pose:
    lines: tuple
    atom_lines: tuple
    serials: tuple
    types: tuple
    branches: tuple
    coords: np.ndarray

    @classmethod
    def parse(cls,text):
        lines=tuple(l for l in text.splitlines() if not l.startswith(('MODEL','ENDMDL','REMARK VINA')))
        serials=[]; atom_lines=[]; types=[]; coords=[]; branches=[]; stack=[]
        for i,l in enumerate(lines):
            if l.startswith(('ATOM  ','HETATM')):
                serial=int(l[6:11]); kind=l[77:].strip()
                if serial in serials:
                    raise ValueError('Duplicate atom serial')
                if kind=='W':
                    raise ValueError('Hydrated docking representation is unsupported')
                serials.append(serial); atom_lines.append(i); types.append(kind)
                coords.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                for b in stack: b[2].append(serial)
            elif l.startswith('BRANCH'):
                _,a,b=l.split(); branch=[int(a),int(b),[]]
                branches.append(branch);stack.append(branch)
            elif l.startswith('ENDBRANCH'):
                _,a,b=l.split()
                if not stack or stack[-1][:2]!=[int(a),int(b)]:
                    raise ValueError('Invalid branch tree')
                stack.pop()
        if stack or not coords or not np.isfinite(coords).all():
            raise ValueError('Unclosed branch or invalid coordinates')
        if len([l for l in lines if l=='ROOT'])!=1:
            raise ValueError('Expected one ligand root')
        for kind in set(types):
            if kind.startswith('CG'):
                if types.count(kind)!=2 or types.count('G'+kind[2:])!=2:
                    raise ValueError('Incomplete macrocycle closure group')
            elif kind.startswith('G') and 'C'+kind not in types:
                raise ValueError('Unmatched macrocycle dummy')
        lookup={s:i for i,s in enumerate(serials)}
        converted=[]
        for a,b,desc in branches:
            if a not in lookup or b not in lookup or b not in desc or a in desc:
                raise ValueError('Invalid branch membership')
            converted.append((lookup[a],lookup[b],tuple(lookup[s] for s in desc)))
        return cls(lines,tuple(atom_lines),tuple(serials),tuple(types),tuple(converted),np.asarray(coords))

    def with_coords(self,coords):
        xyz=np.asarray(coords,dtype=float)
        if xyz.shape!=self.coords.shape or not np.isfinite(xyz).all():
            raise ValueError('Invalid coordinates')
        return Pose(self.lines,self.atom_lines,self.serials,self.types,self.branches,xyz.copy())

    def text(self):
        lines=list(self.lines)
        for i,xyz in zip(self.atom_lines,self.coords):
            formatted=''.join(f'{v:8.3f}' for v in xyz)
            if len(formatted)!=24:
                raise ValueError('PDBQT coordinate overflow')
            lines[i]=lines[i][:30]+formatted+lines[i][54:]
        return '\n'.join(lines)+'\n'

    def torsion(self,index,angle):
        a,b,desc=self.branches[index]
        xyz=self.coords.copy()
        xyz[list(desc)]=rotate(xyz[list(desc)],xyz[b]-xyz[a],angle,xyz[a])
        return self.with_coords(xyz)

    def active_branches(self):
        groups=[{i for i,t in enumerate(self.types) if t in (k,'G'+k[2:])}
                for k in set(self.types) if k.startswith('CG')]
        return [i for i,(_,_,desc) in enumerate(self.branches)
                if all(not (set(desc)&g) or g.issubset(desc) for g in groups)]

    def closure_ok(self,tolerance=.3):
        for i,k in enumerate(self.types):
            if k.startswith('G'):
                partners=[j for j,t in enumerate(self.types) if t=='C'+k]
                if not partners or min(np.linalg.norm(self.coords[i]-self.coords[j]) for j in partners)>tolerance:
                    return False
        return True

    def propose(self,kind,vector,angle,branch,translation=1.,rotation=.35,torsion=.5):
        vector=np.asarray(vector,dtype=float)
        if kind==0:
            return self.with_coords(self.coords+vector*translation)
        active=self.active_branches()
        if kind==1 or not active:
            return self.with_coords(rotate(self.coords,vector,angle*rotation,self.coords.mean(axis=0)))
        return self.torsion(active[branch%len(active)],angle*torsion)

    def randomized(self,rng,center,size):
        from scipy.spatial.transform import Rotation
        xyz=(self.coords-self.coords.mean(axis=0))@Rotation.random(random_state=rng).as_matrix().T
        half=np.asarray(size,dtype=float)/2-.05
        lo=-half-xyz.min(axis=0); hi=half-xyz.max(axis=0)
        if np.any(lo>=hi):
            raise ValueError('Ligand does not fit in search box')
        return self.with_coords(xyz+np.asarray(center)+rng.uniform(lo,hi))

    def inside(self,center,size):
        return bool((np.abs(self.coords-np.asarray(center))<=np.asarray(size)/2-.001).all())
