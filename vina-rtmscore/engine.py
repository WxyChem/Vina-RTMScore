"""Batched multi-start hybrid search. No native coordinates enter this module."""
from dataclasses import asdict
import math
import time
import numpy as np
from vina_rtmscore.objective import Calibration, objective, metropolis
from vina_rtmscore.pose_moves import InvalidPoseError
from vina_rtmscore.hybrid_moves import randomize, mutate


def search(start, center, size, relax, score, config, emit, checkpoint=None):
    chains, warmup, steps, restarts = (config[k] for k in ('chains', 'warmup', 'steps', 'restarts'))
    if min(chains, warmup, steps, restarts) < 1:
        raise ValueError('Search budgets must be positive')
    archive = []; traces = []; serial = 0
    counts = dict(local_calls=0, invalid_proposals=0, neural_poses=0)
    begin = time.monotonic()
    initialization = []

    def log(row):
        row['elapsed_s'] = time.monotonic()-begin
        traces.append(row); emit(row)

    def evaluate(poses, origin, restart, step):
        nonlocal serial
        good = []; outputs = [None] * len(poses); errors = [None] * len(poses)
        for i, pose in enumerate(poses):
            if pose is None:
                continue
            if not pose.inside(center, size) or not pose.closure_ok():
                errors[i] = 'proposal_outside_box_or_open_macrocycle'
                counts['invalid_proposals'] += 1
            else:
                good.append((i, pose))
        counts['local_calls'] += len(good)
        if hasattr(relax, 'map'):
            values = relax.map([p for _, p in good])
        else:
            values = []
            for _, pose in good:
                try:
                    values.append(relax(pose))
                except InvalidPoseError as exc:
                    values.append(exc)
        if len(values) != len(good):
            raise ValueError('Local optimizer returned an incomplete batch')
        for (chain, _), value in zip(good, values):
            if isinstance(value, InvalidPoseError):
                counts['invalid_proposals'] += 1; errors[chain] = str(value); continue
            optimized, energy = value
            if not optimized.inside(center, size) or not optimized.closure_ok():
                counts['invalid_proposals'] += 1; errors[chain] = 'optimized_pose_invalid'; continue
            if not math.isfinite(energy):
                raise ValueError('Nonfinite local energy')
            serial += 1
            state = dict(id=f'p{serial:08d}', pose=optimized, energy=float(energy), rtm=0.,
                         origin=origin, chain=chain, restart=restart, step=step)
            archive.append(state); outputs[chain] = state
            if checkpoint:
                checkpoint(dict(event='evaluated', pose_id=state['id'], origin=origin, chain=chain,
                    restart=restart, step=step, energy=float(energy), coords=optimized.coords.tolist(),
                    elapsed_s=time.monotonic()-begin))
        return outputs, errors

    def neural(states):
        states = [s for s in states if s is not None]
        if not states:
            return
        values = np.asarray(score(states), dtype=float)
        if values.shape != (len(states),) or not np.isfinite(values).all():
            raise ValueError('Invalid neural prediction batch')
        counts['neural_poses'] += len(states)
        for state, value in zip(states, values):
            state['rtm'] = float(value)
            if checkpoint:
                checkpoint(dict(event='scored', pose_id=state['id'], rtm=float(value),
                                elapsed_s=time.monotonic()-begin))

    def initialize(restart, origin, rngs):
        parents = [None] * chains; attempts = [0] * chains
        for attempt in range(20):
            poses = [None] * chains
            active = [chain for chain in range(chains) if parents[chain] is None]
            generation_errors = {}
            for chain in range(chains):
                if parents[chain] is not None:
                    continue
                attempts[chain] += 1
                try:
                    poses[chain] = randomize(start, rngs[chain], center, size)
                except ValueError as exc:
                    if str(exc) != 'Ligand does not fit in search box':
                        raise
                    counts['invalid_proposals'] += 1
                    generation_errors[chain] = str(exc)
            outputs, errors = evaluate(poses, origin, restart, -1)
            for chain in active:
                state = outputs[chain]
                log(dict(phase='initialization', mode='vina_warmup' if origin == 'warmup' else 'hybrid',
                    restart=restart, chain=chain, attempt=attempt+1,
                    candidate_id=state['id'] if state else None, accepted=state is not None,
                    reason=generation_errors.get(chain) or errors[chain],
                    proposed_coords=poses[chain].coords.tolist() if poses[chain] is not None else None))
            for chain, state in enumerate(outputs):
                if state is not None:
                    parents[chain] = state
            if all(p is not None for p in parents):
                break
        else:
            raise ValueError(f'Could not initialize all chains at restart {restart}')
        initialization.append(dict(restart=restart, attempts=attempts,
                                   state_ids=[s['id'] for s in parents]))
        return parents

    def rngs_for(phase, restart):
        return [np.random.default_rng(np.random.SeedSequence([config['seed'], phase, restart, chain]))
                for chain in range(chains)]

    rngs = rngs_for(0, 0)
    parents = initialize(0, 'warmup', rngs)
    for step in range(warmup):
        proposed = [mutate(parent['pose'], rng, config['translation']) for parent, rng in zip(parents, rngs)]
        uniforms = [float(rng.random()) for rng in rngs]
        outputs, errors = evaluate([p for p, _ in proposed], 'warmup', 0, step)
        for chain, (old, candidate, uniform) in enumerate(zip(parents, outputs, uniforms)):
            accepted = candidate is not None and metropolis(old['energy'], candidate['energy'], 1., uniform)
            if accepted:
                parents[chain] = candidate
            log(dict(phase='warmup', mode='vina_warmup', restart=0, chain=chain, step=step,
                parent_id=old['id'], candidate_id=candidate['id'] if candidate else None,
                accepted=accepted, reason=errors[chain], uniform=uniform,
                old_objective=old['energy'], new_objective=candidate['energy'] if candidate else None,
                temperature=1., proposal=proposed[chain][1]))
    neural(archive)
    calibration = Calibration.fit([s['energy'] for s in archive], [s['rtm'] for s in archive])
    warmup_time = time.monotonic()-begin
    warmup_candidates = len(archive)
    initial_parents = [s['id'] for s in parents]
    search_begin = time.monotonic()
    final_states = []
    for restart in range(restarts):
        rngs = rngs_for(1, restart)
        if restart:
            parents = initialize(restart, 'hybrid', rngs)
            neural(parents)
        for step in range(steps):
            proposed = [mutate(parent['pose'], rng, config['translation']) for parent, rng in zip(parents, rngs)]
            uniforms = [float(rng.random()) for rng in rngs]
            outputs, errors = evaluate([p for p, _ in proposed], 'hybrid', restart, step)
            neural(outputs)
            for chain, (old, candidate, uniform) in enumerate(zip(parents, outputs, uniforms)):
                before = objective(old['energy'], old['rtm'], 'hybrid', calibration, config['weight'])
                after = objective(candidate['energy'], candidate['rtm'], 'hybrid', calibration,
                                  config['weight']) if candidate is not None else None
                accepted = candidate is not None and metropolis(before, after, config['temperature'], uniform)
                if accepted:
                    parents[chain] = candidate
                log(dict(phase='search', mode='hybrid', restart=restart, chain=chain, step=step,
                    parent_id=old['id'], candidate_id=candidate['id'] if candidate else None,
                    accepted=accepted, reason=errors[chain], uniform=uniform,
                    energy=candidate['energy'] if candidate else None, rtm=candidate['rtm'] if candidate else None,
                    old_objective=before, new_objective=after, temperature=config['temperature'],
                    proposal=proposed[chain][1]))
        final_states.append([s['id'] for s in parents])
    return archive, dict(calibration=asdict(calibration), warmup_elapsed_s=warmup_time,
        search_elapsed_s=time.monotonic()-search_begin, warmup_candidates=warmup_candidates,
        starting_states=initial_parents, final_states=final_states, restarts=restarts,
        initialization=initialization, counts=counts,
        proposal_budget=chains * steps * restarts, warmup_proposal_budget=chains * warmup,
        initialization_attempt_cap_per_chain=20,
        initialization_attempts=sum(sum(x['attempts']) for x in initialization),
        accepted=sum(t['accepted'] for t in traces if t['phase'] == 'search'))
