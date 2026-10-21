"""Lossless compressed trusted diagnostic snapshots; append-only publication."""
import gzip
import os
from pathlib import Path
import pickle
import tempfile

from planner_evidence import json_write, file_sha256


def save_state(run,name,state):
    target=run.path/'state'/(name+'.pkl.gz')
    with tempfile.NamedTemporaryFile('wb',dir=target.parent,delete=False) as f:
        temporary=Path(f.name)
        try:
            with gzip.GzipFile(fileobj=f,mode='wb',compresslevel=3,mtime=0,filename='') as stream:
                pickle.dump(state,stream,protocol=5)
            f.flush();os.fsync(f.fileno())
        except BaseException:
            temporary.unlink();raise
    try:os.link(temporary,target)
    finally:temporary.unlink()
    json_write(target.with_suffix(target.suffix+'.json'),{'sha256':file_sha256(target),
        'codec':'gzip_pickle5','purpose':'trusted local diagnostic public lifecycle; not training checkpoint'})
    return target


def restore_state(path):
    import json
    path=Path(path);metadata=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if metadata.get('codec')!='gzip_pickle5' or file_sha256(path)!=metadata['sha256']:
        raise ValueError('Diagnostic snapshot codec/checksum mismatch')
    with gzip.open(path,'rb') as stream:return pickle.load(stream)
