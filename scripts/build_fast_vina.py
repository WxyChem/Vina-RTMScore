#!/usr/bin/env python3
"""Build an isolated Vina 1.2.7 C library; installs nothing in any environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://files.pythonhosted.org/packages/d2/2a/6746ef5e57b1c643e9fb24ad9e4fa520add7338736d50954e0fbc12ae52e/vina-1.2.7.tar.gz'
SHA256 = '79e5288d10207b85f20adac3dd6f4708eac761c2d8070fbad76ff997cf48e4f9'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boost-prefix', type=Path, required=True)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    target = ROOT / 'native'
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'vina-1.2.7.tar.gz'
    if not archive.exists():
        raise FileNotFoundError('Bundled Vina source archive is missing: '+str(archive))
    if digest(archive) != SHA256:
        raise ValueError('Official Vina archive hash mismatch')
    original = target / 'vina-1.2.7'
    if not original.exists():
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                resolved = (target / member.name).resolve()
                if not resolved.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                    raise ValueError('Unsafe source archive member')
            source.extractall(target)
    build = target / 'build'
    build.mkdir(exist_ok=True)
    source = build / 'lib'
    shutil.copytree(original / 'src/lib', source, dirs_exist_ok=True)
    header = source / 'vina.h'
    text = header.read_text()
    if text.count('class Vina {') != 1:
        raise ValueError('Unexpected official Vina class declaration')
    header.write_text(text.replace('class Vina {', 'class Vina {\n    friend class FastVina;'))
    flags = ['-O3', '-DNDEBUG', '-std=c++11', '-fPIC', '-DBOOST_ERROR_CODE_HEADER_ONLY',
             '-I' + str(source), '-I' + str(args.boost_prefix / 'include')]
    sources = sorted(source.glob('*.cpp')) + [target / 'fast_vina.cpp']
    from concurrent.futures import ThreadPoolExecutor
    commands = []
    objects = []
    for cpp in sources:
        obj = build / (cpp.stem + '.o')
        command = [args.compiler, *flags, '-c', str(cpp), '-o', str(obj)]
        commands.append(command)
        objects.append(obj)
    start = time.perf_counter()
    def compile_one(command):
        print('Compiling ' + Path(command[-3]).name, flush=True)
        subprocess.run(command, check=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile_one, commands))
    output = target / 'libfast_vina.so'
    link = [args.compiler, '-shared', '-Wl,-Bsymbolic', *map(str, objects),
            '-L' + str(args.boost_prefix / 'lib'),
            '-Wl,-rpath,' + str(args.boost_prefix / 'lib'),
            '-lboost_thread', '-lboost_serialization', '-lboost_filesystem',
            '-lboost_program_options', '-pthread', '-o', str(output)]
    subprocess.run(link, check=True)
    provenance = dict(official_version='1.2.7', archive_url=URL, archive_sha256=SHA256,
        compiler=subprocess.check_output([args.compiler, '--version'], text=True),
        platform=platform.platform(), flags=flags, commands=commands, link_command=link,
        build_seconds=time.perf_counter()-start, library_sha256=digest(output),
        adapter_sha256=digest(target / 'fast_vina.cpp'),
        source_sha256={str(p.relative_to(original)): digest(p)
                       for p in sorted((original / 'src/lib').glob('*')) if p.is_file()},
        patch='Add only friend class FastVina declaration to isolated vina.h; no energy/optimizer changes.')
    (target / 'build_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
