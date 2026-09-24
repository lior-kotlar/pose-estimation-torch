"""Talking to the cluster: one ssh connection per step, and never any work on the gateway.

Everything a PC asks the cluster to do goes through code/local_reanalysis_server.py over ssh, and
every one of those commands is handed to slurm with srun so it runs on a compute node -- the lab's
login gateway is the way in, not a workplace. This module is the transport on its own, so the
programs above it (the re-analysis run and the cluster rounds) share one implementation of it.
"""
import posixpath
import shlex
import subprocess
import json
import hashlib

SERVER_HELPER = 'code/local_reanalysis_server.py'
# A login shell on the gateway has neither slurm on its PATH nor SLURM_CONF in its environment,
# so srun there cannot even find the cluster. Both are named here.
SLURM_BIN = '/vol/slurm/moriah/bindir/bin'
SLURM_CONF = '/vol/slurm/moriah/slurm.conf'


class Problem(Exception):
    """Something the user has to fix; printed without a traceback."""


def on_node(settings, command, attempts=1):
    """`command`, wrapped so that slurm runs it on a compute node.

    The lab's login gateway is for logging in, not for working, so everything this PC asks the
    cluster to do -- reading the declarations, checking a file in, tarring up the code, running
    the helper -- is handed to srun, which queues it and runs it on whichever node is free. srun
    passes stdin and stdout straight through, so the tar streams work as they did; its own
    progress messages go to stderr, where they cannot get into the data. The gateway's shell has
    neither srun on its PATH nor SLURM_CONF set, so both are named before it is called.

    A node sometimes comes up without the lab filesystem mounted (the automount expires under
    load), which would fail a command for no reason of its own. So the node waits for the project
    to appear before starting, and a command that neither reads from this PC nor streams its
    answer back may be given to slurm again, which usually lands it somewhere else."""
    flags = (settings.get('srun_flags') or '').strip()
    if not flags:
        return command
    project = shlex.quote(settings['server_project'])
    wait = (f'P={project}; n=0; while [ ! -d "$P" ] && [ $n -lt 30 ]; do ls -d "$P" >/dev/null 2>&1; '
            'sleep 2; n=$((n+1)); done; '
            'if [ ! -d "$P" ]; then echo "$(hostname) cannot see $P -- either the lab '
            'filesystem is not mounted there, or the server_project setting is wrong" >&2; '
            'exit 75; fi; ')
    launcher = (f'if [ -z "$SLURM_CONF" ] && [ -f {SLURM_CONF} ]; then SLURM_CONF={SLURM_CONF}; '
                'export SLURM_CONF; fi; '
                'if command -v srun >/dev/null 2>&1; then _srun=srun; '
                f'else _srun={SLURM_BIN}/srun; fi; ')
    step = f'"$_srun" {flags} /bin/sh -c {shlex.quote(wait + command)}'
    if attempts > 1:
        # only the unmounted node is worth another node; anything else is the command's own
        # answer and is passed back as it is
        return launcher + (f'for _try in $(seq {attempts}); do {step}; _rc=$?; '
                           '[ $_rc -ne 75 ] && exit $_rc; done; exit $_rc')
    return launcher + step


def remote(settings, command, batch=False, **popen_kwargs):
    """Start `command` in a shell on the cluster, over ssh."""
    if not settings.get('server_user'):
        raise Problem("no server username in the settings; run local_reanalysis\\setup.bat")
    argv = ['ssh', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30']
    if batch:
        argv += ['-o', 'BatchMode=yes']
    argv += [f"{settings['server_user']}@{settings['server_host']}", command]
    try:
        return subprocess.Popen(argv, **popen_kwargs)
    except FileNotFoundError:
        raise Problem("the 'ssh' command was not found. On Windows, turn on 'OpenSSH Client' in "
                      "Settings > System > Optional features, then try again")


def helper_command(settings, *args, attempts=1):
    helper = posixpath.join(settings['server_project'], SERVER_HELPER)
    return on_node(settings, ' '.join(['python3', shlex.quote(helper)]
                                      + [shlex.quote(a) for a in args]), attempts=attempts)


def stage(number, total, text):
    print(f"\n=== {number}/{total}  {text} ===", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def count(items):
    tally = {}
    for item in items:
        tally[item] = tally.get(item, 0) + 1
    return ', '.join(f'{k} {v}' for k, v in sorted(tally.items()))


def server_answer(proc, what):
    """The one JSON line a helper verb prints, or a Problem naming what went wrong.

    When its errors were captured as well -- they are for the short questions, so that slurm's
    queueing messages stay out of the window -- they go into the Problem, since that is where
    'could not reach slurm' would appear."""
    if proc.stderr is not None:
        out, errors = proc.communicate()
        output = out.decode('utf-8', errors='replace').strip()
        errors = errors.decode('utf-8', errors='replace').strip()
    else:
        output = proc.stdout.read().decode('utf-8', errors='replace').strip()
        proc.wait()
        errors = ''
    try:
        answer = json.loads(output.splitlines()[-1])
    except (IndexError, ValueError):
        answer = {'ok': False,
                  'error': ' '.join(part for part in (output, errors) if part)
                           or f'no answer from the server (exit {proc.returncode})'}
    if not answer.get('ok'):
        raise Problem(f"{what}: {answer.get('error')}")
    return answer


def helper_json(settings, *args, what='the server could not do that'):
    # these verbs read nothing from this PC and answer in one line, so slurm may be asked again
    proc = remote(settings, helper_command(settings, *args, attempts=3), stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE)
    return server_answer(proc, what)
