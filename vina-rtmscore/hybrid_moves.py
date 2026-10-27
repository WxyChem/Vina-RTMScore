"""Native-Vina-style global moves with explicit macrocycle closure guards."""
import numpy as np
from vina_rtmscore.pose_moves import rotate


def randomize(start, rng, center, size):
    pose = start
    for branch in pose.active_branches():
        pose = pose.torsion(branch, float(rng.uniform(-np.pi, np.pi)))
    return pose.randomized(rng, center, size)


def inside_sphere(rng):
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    if norm < 1e-15:
        direction = np.array([1., 0., 0.]); norm = 1.
    return direction / norm * float(rng.random() ** (1. / 3))


def mutate(parent, rng, translation=2.):
    """Choose uniformly among translation, orientation and each active torsion.

    Full-period relative torsion is uniform for any current torsion angle, so it
    gives the same global coverage as resampling its absolute angle.
    """
    branches = parent.active_branches()
    entity = int(rng.integers(2 + len(branches)))
    if entity == 0:
        vector = inside_sphere(rng)
        return parent.with_coords(parent.coords + translation * vector), dict(
            entity=entity, kind='translation', vector=vector.tolist(), amplitude=translation)
    if entity == 1:
        vector = inside_sphere(rng)
        origin = parent.coords.mean(axis=0)
        radius = float(np.sqrt(((parent.coords-origin)**2).sum(axis=1).mean()))
        angle = translation * float(np.linalg.norm(vector)) / max(radius, 1e-8)
        result = parent.with_coords(rotate(parent.coords, vector, angle, origin))
        return result, dict(entity=entity, kind='rotation', vector=vector.tolist(),
                            angle=angle, gyration_radius=radius)
    branch = branches[entity-2]
    angle = float(rng.uniform(-np.pi, np.pi))
    return parent.torsion(branch, angle), dict(entity=entity, kind='torsion', branch=branch, angle=angle)
