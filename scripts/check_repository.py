"""Check local Markdown links and package/runtime/README version consistency."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
failures = []
links = 0
for path in root.rglob('*.md'):
    if any(part.startswith('.') or part in ('build', 'outputs') for part in path.relative_to(root).parts):
        continue
    text = path.read_text(encoding='utf-8')
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
        if re.match(r'[a-z]+:', target) or target.startswith('#'):
            continue
        target = target.split('#', 1)[0].split(' "', 1)[0]
        resolved = (path.parent / target).resolve()
        links += 1
        if not resolved.is_relative_to(root) or not resolved.exists():
            failures.append(f'{path.relative_to(root)}: missing/outside link {target}')
version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(encoding='utf-8'), re.M).group(1)
if f'__version__ = "{version}"' not in (root/'cvd_cbd/__init__.py').read_text(encoding='utf-8'):
    failures.append('Runtime/package version mismatch')
for name in ('README.md', 'README.ko.md'):
    if version not in (root/name).read_text(encoding='utf-8').splitlines()[0]:
        failures.append(f'{name}: title version mismatch')
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: {links} local Markdown links; package/runtime/README version {version}')
