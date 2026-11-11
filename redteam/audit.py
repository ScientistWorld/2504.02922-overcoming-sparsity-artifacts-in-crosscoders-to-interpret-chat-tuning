"""Executable benchmark audit. Never imports the hidden evaluator.

Every attempt installs ordinary artifacts or data as a future agent could,
then launches the unmodified hidden shell entry point with NO arguments.
The original artifacts are reused only for checks that need no retraining.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
STATE = AUDIT / '_state'
ARTIFACTS = ROOT / 'checkpoints/crosscoder_synthetic'
FILES = ['scoring/scores_train.json', 'scoring/scores_test.json',
         'data/synthetic_activations.npz', 'data/synthetic_metadata.json']
EXPERIMENT = 'synthetic_chat_specificity'

# These describe actual procedures, not desired results. No test-score tuning.
CONFIGS = {
    'reference_saved': {'kind': 'saved'},
    'l1_baseline': {'kind': 'baseline'},
    **{f'reference_seed{s}': {'kind': 'train', 'epochs': 80, 'seed': s}
       for s in (1, 2, 3)},
    **{f'better_longer_seed{s}': {'kind': 'train', 'epochs': 320, 'seed': s}
       for s in (1, 2, 3)},
    **{f'better_amplitude_seed{s}': {'kind': 'amplitude', 'seed': s}
       for s in (1, 2, 3)},
    **{f'weaker_short_seed{s}': {'kind': 'train', 'epochs': 2, 'seed': s}
       for s in (1, 2, 3)},
    'pca_baseline': {'kind': 'pca'},
    'brute_force_exemplars': {'kind': 'brute'},
    'shortcut_visible_labels': {'kind': 'oracle'},
    'shortcut_zero_outputs': {'kind': 'zero'},
    'shortcut_empty_features': {'kind': 'empty_features'},
    'shortcut_copy_reconstruction': {'kind': 'copy'},
    'shortcut_threshold_mask': {'kind': 'threshold'},
    'shortcut_negative_features': {'kind': 'negative'},
    'shortcut_nan_reconstruction': {'kind': 'nan'},
    'shortcut_taxonomy_replacement': {'kind': 'taxonomy'},
    'shortcut_target_replacement': {'kind': 'targets'},
    'shortcut_score_write': {'kind': 'score_write'},
    'shortcut_registry_omission': {'kind': 'omit'},
    'shortcut_unlisted_method': {'kind': 'unlisted'},
    'shortcut_missing_artifact': {'kind': 'missing'},
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + '\n')


def initialize():
    if (STATE / 'ready').exists():
        return
    STATE.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        dest = STATE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    shutil.copytree(ARTIFACTS, STATE / 'artifacts', dirs_exist_ok=True)
    dump(AUDIT / 'reference_scores_original.json',
         json.loads((ROOT / 'scoring/scores_test.json').read_text()))
    (STATE / 'ready').touch()


def restore():
    if not (STATE / 'ready').exists():
        return
    for rel in FILES:
        shutil.copy2(STATE / rel, ROOT / rel)
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    shutil.copytree(STATE / 'artifacts', ARTIFACTS)


def load_npz(path):
    import numpy as np
    with np.load(path) as src:
        return {k: src[k].copy() for k in src.files}


def build(name, config, folder):
    import numpy as np
    kind = config['kind']
    path = ARTIFACTS / 'batchtopk' / 'val_outputs.npz'
    out = load_npz(path)
    data_path = ROOT / 'data/synthetic_activations.npz'
    data = load_npz(data_path)
    base, chat = data['val_base'], data['val_chat']
    n, d = base.shape
    meta = json.loads((ROOT / 'data/synthetic_metadata.json').read_text())
    if kind == 'train':
        import torch
        from method.train import train_one
        torch.set_num_threads(1)
        artifact_dir = folder / 'artifacts'
        metrics = train_one('batchtopk', data['train_base'], data['train_chat'],
                            base, chat, artifact_dir, 88, config['epochs'],
                            256, 1e-3, .005, 14, config['seed'], 'cpu')
        shutil.copytree(artifact_dir / 'batchtopk', ARTIFACTS / 'batchtopk',
                        dirs_exist_ok=True)
        dump(folder / 'training.json', metrics)
        return
    if kind == 'saved':
        return
    if kind == 'baseline':
        out = load_npz(ARTIFACTS / 'l1/val_outputs.npz')
    elif kind == 'amplitude':
        # Honest inference improvement: keep the trained dictionary and the
        # encoder's sparse support; optimize only nonnegative amplitudes on
        # the observed input. No coefficients, labels or hidden split are read.
        source = AUDIT / f"reference_seed{config['seed']}/artifacts/batchtopk/val_outputs.npz"
        if not source.exists():
            import torch
            from method.train import train_one
            torch.set_num_threads(1)
            train_one('batchtopk', data['train_base'], data['train_chat'],
                      base, chat, source.parent.parent, 88, 80, 256,
                      1e-3, .005, 14, config['seed'], 'cpu')
        out = load_npz(source)
        decoder = np.concatenate([out['base_decoder'], out['chat_decoder']], axis=1)
        bias = np.concatenate([out['base_bias'], out['chat_bias']])
        target = np.concatenate([base, chat], axis=1) - bias
        f = out['features'].copy()
        support = f > 0
        gram = decoder @ decoder.T
        step = 1. / max(float(np.linalg.eigvalsh(gram).max()), 1e-8)
        target_dot = target @ decoder.T
        initial_mse = float(np.mean((f @ decoder - target) ** 2))
        for _ in range(300):
            f = np.maximum(f - step * (f @ gram - target_dot), 0) * support
        recon = f @ decoder + bias
        out.update(features=f, base_recon=recon[:, :d], chat_recon=recon[:, d:])
        dump(folder / 'inference_fit.json', {
            'source': str(source.relative_to(ROOT)), 'steps': 300,
            'uses_labels': False, 'initial_mse': initial_mse,
            'final_mse': float(np.mean((f @ decoder - target) ** 2)),
            'new_support_entries': int(np.sum((f > 0) & ~support)),
        })
    elif kind == 'pca':
        # A simple joint linear factor baseline, with signed factors encoded
        # using nonnegative positive/negative pairs. Recons match the factors.
        x = np.concatenate([data['train_base'], data['train_chat']], axis=1)
        mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        vectors = vt[:14]
        values = (np.concatenate([base, chat], axis=1) - mean) @ vectors.T
        f = np.concatenate([np.maximum(values, 0), np.maximum(-values, 0)], axis=1)
        dec = np.concatenate([vectors, -vectors], axis=0)
        recon = f @ dec + mean
        out = dict(features=f, base_decoder=dec[:, :d], chat_decoder=dec[:, d:],
                   base_recon=recon[:, :d], chat_recon=recon[:, d:],
                   base_bias=mean[:d], chat_bias=mean[d:])
    elif kind == 'brute':
        # Enumerate every observed training example as a paired dictionary
        # atom; find the best single nonnegative atom by exhaustive search.
        x = np.concatenate([data['train_base'], data['train_chat']], axis=1)
        norms = np.linalg.norm(x, axis=1)
        atoms = x / np.maximum(norms[:, None], 1e-8)
        y = np.concatenate([base, chat], axis=1)
        dots = y @ atoms.T
        idx = np.argmax(dots, axis=1)
        f = np.zeros((n, len(x)), np.float32)
        f[np.arange(n), idx] = np.maximum(dots[np.arange(n), idx], 0)
        recon = f @ atoms
        out = dict(features=f, base_decoder=atoms[:, :d], chat_decoder=atoms[:, d:],
                   base_recon=recon[:, :d], chat_recon=recon[:, d:])
    elif kind == 'oracle':
        # SHORTCUT: supervised access to the supposedly hidden factor answers.
        # Fit two decoders to disclosed ground-truth coefficients. Neither
        # validation-half indices nor any hidden evaluator code are used.
        train_features = data['train_coefficients']
        f = data['val_coefficients']
        design = np.column_stack([train_features, np.ones(len(train_features))])
        db = np.linalg.lstsq(design, data['train_base'], rcond=None)[0]
        dc = np.linalg.lstsq(design, data['train_chat'], rcond=None)[0]
        out = dict(features=f, base_decoder=db[:-1], chat_decoder=dc[:-1],
                   base_recon=f @ db[:-1] + db[-1],
                   chat_recon=f @ dc[:-1] + dc[-1],
                   base_bias=db[-1], chat_bias=dc[-1])
        # Also demonstrate that the public deterministic generator recreates
        # the full validation labels and directions, even if fields are stripped.
        from data.synthetic_activations import generate_dataset
        regen_dir = folder / 'prepared/regenerated'
        generate_dataset(regen_dir, **{k: meta[k] for k in (
            'seed', 'scenario', 'n_train', 'n_val', 'd_model', 'activation_rate',
            'base_weak_scale', 'decoupled_shared_rate', 'decoupled_exclusive_rate',
            'noise_std')})
        regenerated = load_npz(regen_dir / 'synthetic_activations.npz')
        dump(folder / 'label_rebuild.json',
             {k: bool(np.array_equal(v, regenerated[k])) for k, v in data.items()})
    elif kind in ('zero', 'nan'):
        # SHORTCUT: no learned factors, no reconstruction. The empty scaling
        # check treats 0/0 as evidence of specificity, and l0/counts are ideal.
        out = dict(features=np.zeros((n, 1), np.float32),
                   base_decoder=np.zeros((1, d), np.float32),
                   chat_decoder=np.zeros((1, d), np.float32),
                   base_recon=np.full_like(base, np.nan) if kind == 'nan' else np.zeros_like(base),
                   chat_recon=np.full_like(chat, np.nan) if kind == 'nan' else np.zeros_like(chat))
    elif kind == 'empty_features':
        # SHORTCUT: preserve the reference reconstruction/directions, claim
        # that no factors activated. Evaluation never recomputes reconstruction.
        out['features'][:] = 0
    elif kind == 'copy':
        # SHORTCUT: submit the input itself as the reconstruction.
        out['base_recon'] = base.copy()
        out['chat_recon'] = chat.copy()
    elif kind == 'negative':
        # SHORTCUT: invalid signed activations make l0 zero. Their beta
        # numerators flip sign while ratio denominators are forced positive.
        out['features'] *= -1
    elif kind == 'threshold':
        # SHORTCUT: change only the submitted norm-based operating point.
        # q -> .21 + .79*q strictly preserves rank, but prevents delta >= .9.
        bnorm = np.linalg.norm(out['base_decoder'], axis=1)
        cnorm = np.linalg.norm(out['chat_decoder'], axis=1)
        direction = out['base_decoder'] / np.maximum(bnorm[:, None], 1e-12)
        dead = bnorm < 1e-12
        direction[dead] = out['chat_decoder'][dead] / np.maximum(cnorm[dead, None], 1e-12)
        out['base_decoder'] = direction * (.21 * cnorm + .79 * bnorm)[:, None]
    elif kind == 'taxonomy':
        # SHORTCUT: move the ground-truth side of the comparison.
        meta['concept_indices']['shared'] = []
        meta['concept_indices']['weak_shared'] = []
        meta['concept_indices']['chat_only'] = list(range(len(data['directions'])))
        dump(ROOT / 'data/synthetic_metadata.json', meta)
        return
    elif kind == 'targets':
        # SHORTCUT: replace the supposedly fixed validation target artifact
        # with the unchanged reference reconstruction; training stays untouched.
        data['val_base'] = out['base_recon'].copy()
        data['val_chat'] = out['chat_recon'].copy()
        np.savez_compressed(data_path, **data)
        return
    elif kind in ('score_write', 'omit', 'unlisted', 'missing'):
        return
    else:
        raise ValueError(kind)
    np.savez_compressed(path, **out)


def run_command(command, logfile):
    with logfile.open('w') as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def run_one(name):
    initialize()
    restore()
    folder = AUDIT / name
    folder.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[name]
    dump(folder / 'config.json', config)
    start = time.time()
    try:
        build(name, config, folder)
        # Normal public evaluation discovers installed methods; method names
        # are not the audit's choosing at hidden-evaluation time.
        train_rc = run_command(['bash', 'scripts/evaluate_train.sh'], folder / 'train.log')
        if train_rc != 0:
            raise RuntimeError(f'public evaluation failed ({train_rc})')
        kind = config['kind']
        train_file = ROOT / 'scoring/scores_train.json'
        if kind == 'score_write':
            scores = json.loads(train_file.read_text())
            scores['experiments'][EXPERIMENT]['results']['batchtopk'] = {
                'fraction_top_norm_true_chat': 1e9, 'reconstruction_fve': 1e9,
                'mean_l0': -1e9, 'false_ft_only_rate_pi06': None}
            dump(train_file, scores)
            dump(ROOT / 'scoring/scores_test.json', scores)
        if kind == 'omit':
            # SHORTCUT: a failed/no-output method removes all required names.
            shutil.rmtree(ARTIFACTS / 'batchtopk')
            (ARTIFACTS / 'shortcut_no_output').mkdir()
            dump(train_file, {'experiments': {EXPERIMENT: {'results': {}}}})
        if kind == 'unlisted':
            # SHORTCUT: outputs on disk absent from the editable public registry
            # are silently skipped. No hidden evaluator code is modified.
            extra = ARTIFACTS / 'shortcut_unlisted'
            extra.mkdir()
            shutil.copy2(ARTIFACTS / 'batchtopk/val_outputs.npz', extra / 'val_outputs.npz')
        if kind == 'missing':
            shutil.rmtree(ARTIFACTS / 'batchtopk')
        shutil.copy2(train_file, folder / 'scores_train.json')
        # Normally remove stale output so a crash cannot look successful.
        # The score-writing shortcut deliberately retains its forged file to
        # test whether the real hidden entry point overwrites it.
        if kind != 'score_write':
            (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        methods_on_disk = sorted(p.name for p in ARTIFACTS.iterdir()
                                 if (p / 'val_outputs.npz').exists())
        code = run_command(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        score_path = ROOT / 'scoring/scores_test.json'
        if score_path.exists():
            shutil.copy2(score_path, folder / 'scores_test.json')
        else:
            (folder / 'scores_test.json').unlink(missing_ok=True)
        dump(folder / 'status.json', {
            'hidden_command': 'bash scripts/evaluate_test.sh',
            'hidden_arguments': [], 'returncode': code,
            'fresh_scores_written': score_path.exists(),
            'methods_on_disk': methods_on_disk,
            'prewritten_test_score': kind == 'score_write',
            'seconds': time.time() - start,
        })
        if code:
            (folder / 'failure.txt').write_text((folder / 'test.log').read_text()[-6000:])
        print(f'{name}: hidden_exit={code}; elapsed={time.time()-start:.2f}s', flush=True)
    finally:
        restore()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'all', 'restore'))
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.command == 'restore':
        restore()
    elif args.command == 'run':
        run_one(args.name)
    else:
        for name in CONFIGS:
            try:
                run_one(name)
            except Exception as exc:
                dump(AUDIT / name / 'build_failure.json', {'error': repr(exc)})
                print(f'{name}: FAILED: {exc}', flush=True)


if __name__ == '__main__':
    main()
