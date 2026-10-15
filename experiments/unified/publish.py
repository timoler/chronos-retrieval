# -*- coding: utf-8 -*-
"""Publish only run status/results; never force-push or stage unrelated work."""
import json, os, subprocess, time
from pathlib import Path

def publish(repo, artifacts, output, state, phase=None):
    if os.environ.get('PAER_PUBLISH_PROGRESS') != '1':
        return
    marker = artifacts / 'PUBLISH_STATUS.json'
    previous = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
    changed = (state, phase) != (previous.get('state'), previous.get('phase'))
    if not changed and time.time() - previous.get('attempt_epoch', 0) < 300:
        return
    report = dict(state=state, phase=phase, attempt_epoch=time.time(), ok=False)
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    git = ['git', '--git-dir=' + str(repo / '.git'), '--work-tree=' + str(repo)]
    def run(*args, allow=()):
        proc = subprocess.run(git + list(args), cwd=repo, env=env,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=90,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode != 0 and proc.returncode not in allow:
            raise RuntimeError('git ' + args[0] + ': ' + proc.stderr[-1500:])
        return proc
    try:
        # Refuse to commit someone else's staged changes in this isolated checkout.
        staged = run('diff', '--cached', '--name-only').stdout.splitlines()
        assert all(p == 'docs/unified/STATUS.json' or p.startswith('docs/unified/results/') for p in staged), 'Unrelated staged changes'
        paths = ['docs/unified/STATUS.json']
        if (repo / 'docs/unified/results').exists():
            paths.append('docs/unified/results')
        run('add', '--', *paths)
        if run('diff', '--cached', '--quiet', allow=(1,)).returncode == 1:
            run('commit', '-m', 'Update unified PAER run progress: ' + state + ' ' + str(phase or ''))
        run('push', 'origin', 'HEAD:refs/heads/unified-paer-20261014', 'HEAD:refs/heads/main')
        report.update(ok=True, commit=run('rev-parse', 'HEAD').stdout.strip())
    except Exception as exc:
        # A remote change/auth failure never changes an experiment or triggers force-push.
        report['error'] = str(exc)
        print('PROGRESS PUBLISH FAILED:', str(exc), flush=True)
    for target in [marker, output / 'REPOSITORY_STATUS.json']:
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(target)
