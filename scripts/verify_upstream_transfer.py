#!/usr/bin/env python3
"""Read-only verification of a transferred source/data package, never training."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root,manifest):
    errors=[]
    for relative,record in manifest['files'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Package path escapes destination')
        if not p.is_file():errors.append({'file':relative,'reason':'missing'});continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:errors.append({'file':relative,'reason':'checksum'})
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--root',default='.')
    a=p.parse_args();m=json.loads(Path(a.manifest).read_text());errors=verify(Path(a.root),m)
    print(json.dumps({'files':len(m['files']),'errors':errors,'formal_training_ready':False},indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
