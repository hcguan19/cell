"""Read-only syntax, source provenance and public-release checks.

Use --original-zip with the private source-only retrieval archive to additionally
prove comments did not change the original AST. Never publish that archive.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_ast(data):
    return ast.dump(ast.parse(data), include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-zip', type=Path)
    args = parser.parse_args()
    original = json.loads((ROOT / 'source_manifest.json').read_text())
    annotated = json.loads((ROOT / 'annotated_manifest.json').read_text())
    published = {row['path']: row for row in annotated}
    archive = ZipFile(args.original_zip) if args.original_zip else None
    try:
        for entry in original:
            path = entry['path']
            data = (ROOT / path).read_bytes()
            structure = normalized_ast(data)
            assert digest(data) == published[path]['sha256'], path
            # ast.dump output changes between Python minor versions (for example,
            # Python 3.8 and 3.12 represent parts of the same parsed source
            # differently).  The byte hash above is the portable integrity check;
            # parsing still proves syntax validity on the current interpreter.
            # Exact AST equivalence remains enforced below when the private
            # original archive is explicitly supplied on the same interpreter.
            if archive:
                source = archive.read(path)
                assert digest(source) == entry['sha256'], path
                assert normalized_ast(source) == structure, f'algorithm changed: {path}'
        assert len(original) == len(annotated) == 23
        for p in ROOT.rglob('*.py'):
            if '.git' not in p.parts:
                ast.parse(p.read_bytes(), filename=str(p))
        forbidden = {'.csv', '.tsv', '.xlsx', '.xls', '.npy', '.npz', '.pt', '.pth',
                     '.pyth', '.safetensors', '.mp4', '.avi', '.pkl', '.joblib', '.zip'}
        # Public tree contains explicit source/docs/test files, never patient artifacts.
        for p in ROOT.rglob('*'):
            if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:
                continue
            assert p.suffix.lower() not in forbidden, f'forbidden artifact: {p}'
            assert p.stat().st_size < 1_000_000, f'unexpected large file: {p}'
            if p.name != 'check_release.py':
                value = p.read_text(encoding='utf-8-sig')
                assert not re.search(r'-----BEGIN [A-Z ]*PRIVATE KEY-----', value), p
                assert not re.search(r'(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{25,}', value), p
        print('PASS: 23 source files, syntax, hashes, and release artifact checks')
        if archive:
            print('PASS: exact AST equivalence to retrieved server snapshot')
    finally:
        if archive:
            archive.close()


if __name__ == '__main__':
    main()
