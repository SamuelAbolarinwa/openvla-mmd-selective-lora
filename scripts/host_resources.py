"""Available host RAM bounded by the process's cgroup and ancestor limits."""
from pathlib import Path, PurePosixPath
import re


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def _remaining(directory, version):
    maximum, current = ('memory.max', 'memory.current') if version == 2 else (
        'memory.limit_in_bytes', 'memory.usage_in_bytes')
    try:
        limit = (directory / maximum).read_text().strip()
        if limit == 'max':
            return None
        return max(0, int(limit) - int((directory / current).read_text().strip()))
    except (OSError, ValueError):
        return None


def available_memory_bytes(meminfo=Path('/proc/meminfo'), cgroup=Path('/sys/fs/cgroup'),
                           proc_cgroup=Path('/proc/self/cgroup'),
                           mountinfo=Path('/proc/self/mountinfo')):
    try:
        available = int(next(line.split()[1] for line in meminfo.read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
    except (OSError, StopIteration, ValueError):
        return 0
    candidates = [(cgroup, cgroup, 2), (cgroup/'memory', cgroup/'memory', 1)]
    try:
        memberships = []
        for line in proc_cgroup.read_text().splitlines():
            _, controllers, member = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                memberships.append((2 if not controllers else 1, PurePosixPath(member)))
        for line in mountinfo.read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields, tail = before.split(), after.split()
            version = 2 if tail[0] == 'cgroup2' else 1 if tail[0] == 'cgroup' else 0
            if not version or (version == 1 and 'memory' not in tail[2].split(',')):
                continue
            mount_root = PurePosixPath(_unescape(fields[3]))
            mount = Path(_unescape(fields[4]))
            for member_version, member in memberships:
                if member_version != version:
                    continue
                try:
                    relative = member.relative_to(mount_root)
                except ValueError:
                    # A cgroup namespace can expose '/' while the mount root is a host path.
                    if member == PurePosixPath('/'):
                        relative = PurePosixPath('.')
                    else:
                        continue
                if '..' in relative.parts:
                    continue
                candidates.append((mount / str(relative), mount, version))
    except (OSError, ValueError, IndexError):
        # Fall back to visible root limits if proc metadata is unavailable.
        pass
    seen = set()
    for leaf, mount, version in candidates:
        directory = leaf
        while directory == mount or mount in directory.parents:
            key = (directory, version)
            if key not in seen:
                seen.add(key)
                remaining = _remaining(directory, version)
                if remaining is not None:
                    available = min(available, remaining)
            if directory == mount:
                break
            directory = directory.parent
    return available
