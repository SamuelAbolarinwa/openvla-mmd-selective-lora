"""Respect host and cgroup memory limits when scheduling independent models."""
from pathlib import Path


def available_memory_bytes(meminfo=Path('/proc/meminfo'), cgroup=Path('/sys/fs/cgroup')):
    try:
        available=int(next(line.split()[1] for line in meminfo.read_text().splitlines()
                           if line.startswith('MemAvailable:')))*1024
    except (OSError, StopIteration, ValueError):
        return 0
    for maximum, current in [('memory.max','memory.current'),
                              ('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes')]:
        try:
            limit=(cgroup/maximum).read_text().strip()
            if limit != 'max':
                used=int((cgroup/current).read_text().strip())
                available=min(available,max(0,int(limit)-used))
        except (OSError, ValueError):
            continue
    return available
