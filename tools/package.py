"""Build a submission zip from an engine directory.

    python tools/package.py engines/me submissions/me.zip

Takes the root .py files and weights/ (net plus syzygy tables), nothing else:
no __pycache__, no local tooling. The zip unpacks to agent.py at its root,
which is what the match workflow and the platform both expect.
"""
import os
import sys
import zipfile


def main():
    src, out = sys.argv[1], sys.argv[2]
    names = []
    for entry in sorted(os.listdir(src)):
        if entry.endswith(".py"):
            names.append(entry)
    for root, _, files in os.walk(os.path.join(src, "weights")):
        for f in sorted(files):
            names.append(os.path.relpath(os.path.join(root, f), src))
    assert "agent.py" in names, "no agent.py"
    assert os.path.join("weights", "net.npz") in names, "no net"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(src, n), n.replace(os.sep, "/"))
    size = os.path.getsize(out) / 1e6
    print(f"{out}: {len(names)} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
