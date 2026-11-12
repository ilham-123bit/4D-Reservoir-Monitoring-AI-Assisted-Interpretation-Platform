from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoir_time_machine.data_ingest.cache_identity import paired_cache_provenance
from reservoir_time_machine.processing.full_field import _canonical, run_full_field_attributes
from scripts.hpc.verify_full_field_attributes import verify


class FullFieldIndependentVerifierTests(unittest.TestCase):
    def test_verifies_synthetic_full_coverage_outputs_journal_and_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir(mode=0o700)
            shape=(5,4,13);rng=np.random.default_rng(9)
            base=rng.normal(size=shape).astype(np.float32)
            monitor=(base+rng.normal(scale=.1,size=shape)).astype(np.float32)
            missing=np.zeros(shape[:2],dtype=bool);missing[2,1]=True
            for name,array in [('baseline',base),('monitor',monitor),('missing_trace_mask',missing)]:
                np.save(cache/f'{name}.npy',array)
            metadata={'shape':list(shape),'inline_ids':[100+i for i in range(shape[0])],
                'crossline_ids':[200+i for i in range(shape[1])],
                'paired_trace_count':int((~missing).sum())}
            (cache/'metadata.json').write_text(json.dumps(metadata))
            cache_id,cache_details=paired_cache_provenance(cache)
            details={'schema':'rtm-full-field-input-identity-2','paired_cache_sha256':cache_id,
                'paired_cache':cache_details,'source_bounds':[0,5,0,4,0,13],
                'source_domain_shape':list(shape)}
            identity=hashlib.sha256(json.dumps(details,sort_keys=True,separators=(',',':'),
                allow_nan=False).encode()).hexdigest()
            result=run_full_field_attributes(base,monitor,missing,root/'attributes',
                source_identity=identity,source_identity_details=details,
                chunk_shape=(64,64,256),window_samples=9,min_support_fraction=.8,
                source_bounds=(0,5,0,4,0,13),source_domain_shape=shape,
                protected_cache=cache)
            checked=verify(cache,Path(result['run_directory']))
            self.assertEqual(checked['status'],'verified')
            self.assertEqual(checked['chunks_verified'],1)
            self.assertEqual(checked['output_files_verified'],7)
            self.assertEqual(checked['unoccupied_grid_cells'],1)

    def test_rejects_changed_input_or_tampered_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir(mode=0o700)
            shape=(5,4,13);base=np.ones(shape,dtype=np.float32);monitor=base+1
            missing=np.zeros(shape[:2],dtype=bool)
            for name,array in [('baseline',base),('monitor',monitor),('missing_trace_mask',missing)]:
                np.save(cache/f'{name}.npy',array)
            (cache/'metadata.json').write_text(json.dumps({'shape':list(shape),
                'inline_ids':list(range(1,6)),'crossline_ids':list(range(11,15)),
                'paired_trace_count':20}))
            cache_id,cache_details=paired_cache_provenance(cache)
            details={'schema':'rtm-full-field-input-identity-2','paired_cache_sha256':cache_id,
                'paired_cache':cache_details,'source_bounds':[0,5,0,4,0,13],
                'source_domain_shape':list(shape)}
            identity=hashlib.sha256(json.dumps(details,sort_keys=True,separators=(',',':'),
                allow_nan=False).encode()).hexdigest()
            result=run_full_field_attributes(base,monitor,missing,root/'attributes',
                source_identity=identity,source_identity_details=details,chunk_shape=(64,64,256),
                source_bounds=(0,5,0,4,0,13),source_domain_shape=shape,protected_cache=cache)
            run=Path(result['run_directory'])
            with (run/'signed_difference.npy').open('r+b') as stream:
                stream.seek(-1,2);value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                verify(cache,run)


if __name__=='__main__':unittest.main()
