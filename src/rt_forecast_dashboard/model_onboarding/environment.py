"""Match the declared complete runtime lock to the environment that actually ran."""
import re


def canonical(name):
    return re.sub(r'[-_.]+','-',name).lower()


def locked_environment(lines, worker):
    declared=set()
    versions=worker.get('active_versions',{})
    commits=worker.get('source_revisions',{})
    if not versions:
        return False
    for line in lines:
        pin=re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^ ;]+)',line)
        vcs=re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*@\s*git\+\S+@([a-fA-F0-9]{40})(?:#\S+)?',line)
        if pin:
            name,version=pin.groups();name=canonical(name)
            if versions.get(name)!=version or commits.get(name): return False
        elif vcs:
            name,revision=vcs.groups();name=canonical(name)
            if commits.get(name)!=revision: return False
        else:
            return False
        declared.add(name)
    # Venv bootstrap tools may be supplied by the selected Python distribution.
    return bool(declared) and set(versions)-declared <= {'pip','setuptools','wheel'}
