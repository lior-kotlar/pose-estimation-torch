"""The cluster side of local_reanalysis.py, run over ssh with the system python3.

local_reanalysis.py (on a PC) talks to the cluster only by running this script over ssh and
piping data through it, so the whole exchange is one connection per step and needs nothing on
the cluster beyond this file and a python3. Standard library only, on purpose: the login node
runs it with whatever python3 it has, not the project's .env.

    python3 local_reanalysis_server.py declarations
        stdin : a JSON list of perturbation.json paths the PC's movies may need
        stdout: a tar of the ones that exist, each stored under its absolute path minus the
                leading '/'

    python3 local_reanalysis_server.py receive --dest <collected_h5 dir>
        stdin : a tar whose first member is MANIFEST.json ({"files": {relpath: sha256}}),
                followed by exactly those files
        stdout: one JSON line, {"ok": true, "stamp": ..., "results": {relpath: status}}

receive unpacks into a hidden staging folder inside <dest>, checks every file's sha256 against
the manifest, and only then installs them: an identical file already in place is 'unchanged',
a different one is moved into superseded_<stamp>/ next to it ('replaced'), a missing one is
'new'. Nothing is installed unless every file arrived intact.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time

MANIFEST = "MANIFEST.json"
DECLARATION_NAME = "perturbation.json"
DIR_MODE = 0o2755   # setgid, so new folders keep the project's group like the rest of the tree
FILE_MODE = 0o644


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relpath(name):
    """A tar member name that stays inside the folder it is unpacked into."""
    parts = name.replace("\\", "/").split("/")
    return (bool(name) and not name.startswith("/") and ".." not in parts
            and all(parts) and ":" not in parts[0])


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        try:
            os.chmod(path, DIR_MODE)
        except OSError:
            pass


def declarations():
    wanted = json.load(sys.stdin)
    out = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    sent = set()
    for path in wanted:
        path = os.path.normpath(str(path))
        if (os.path.basename(path) != DECLARATION_NAME or not os.path.isabs(path)
                or path in sent or not os.path.isfile(path)):
            continue
        out.add(path, arcname=path.lstrip("/"), recursive=False)
        sent.add(path)
    out.close()
    sys.stdout.buffer.flush()
    return 0


def receive(dest):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    makedirs(dest)
    staging = os.path.join(dest, ".incoming_%s_%d" % (stamp, os.getpid()))
    os.makedirs(staging)
    try:
        manifest = None
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
            for member in tar:
                if member.name == MANIFEST:
                    manifest = json.loads(tar.extractfile(member).read().decode("utf-8"))["files"]
                    continue
                if manifest is None:
                    raise ValueError("the upload did not start with %s" % MANIFEST)
                if not member.isfile() or not safe_relpath(member.name) or member.name not in manifest:
                    raise ValueError("unexpected entry in the upload: %r" % member.name)
                target = os.path.join(staging, *member.name.split("/"))
                if not os.path.isdir(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                source = tar.extractfile(member)
                with open(target, "wb") as f:
                    shutil.copyfileobj(source, f)
        if manifest is None:
            raise ValueError("empty upload")

        for rel, digest in manifest.items():
            staged = os.path.join(staging, *rel.split("/"))
            if not os.path.isfile(staged):
                raise ValueError("%s is in the manifest but was not received" % rel)
            if sha256(staged) != digest:
                raise ValueError("%s arrived damaged (checksum mismatch)" % rel)

        results = {}
        for rel in sorted(manifest):
            staged = os.path.join(staging, *rel.split("/"))
            final = os.path.join(dest, *rel.split("/"))
            if os.path.isfile(final):
                if sha256(final) == manifest[rel]:
                    results[rel] = "unchanged"
                    continue
                archive = os.path.join(os.path.dirname(final), "superseded_%s" % stamp)
                makedirs(archive)
                os.replace(final, os.path.join(archive, os.path.basename(final)))
                results[rel] = "replaced"
            else:
                results[rel] = "new"
            parent = os.path.dirname(final)
            missing = []
            while parent and not os.path.isdir(parent):
                missing.append(parent)
                parent = os.path.dirname(parent)
            for folder in reversed(missing):
                makedirs(folder)
            os.replace(staged, final)
            os.chmod(final, FILE_MODE)
        print(json.dumps({"ok": True, "stamp": stamp, "dest": dest, "results": results}))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("declarations")
    rec = sub.add_parser("receive")
    rec.add_argument("--dest", required=True)
    args = parser.parse_args()
    if args.command == "declarations":
        return declarations()
    if args.command == "receive":
        return receive(args.dest)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
