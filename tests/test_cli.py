import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

class CliTests(unittest.TestCase):
    def test_cli_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            input_path=root/'input.tif'
            output=root/'output'
            with rasterio.open(input_path,'w',driver='GTiff',height=20,width=20,count=1,
                dtype='float32',nodata=-9999,crs='EPSG:32648',
                transform=from_origin(400000,2000000,10,10)) as ds:
                ds.write(np.ones((20,20),dtype=np.float32),1)
            output.mkdir()
            (output/'run.json').write_text('{}',encoding='utf8')
            cmd=[sys.executable,'-m','gslat.cli','--input',str(input_path),
                '--output-dir',str(output),'--representation','positive']
            env=os.environ.copy()
            completed=subprocess.run(cmd,capture_output=True,text=True,env=env)
            self.assertNotEqual(completed.returncode,0)
            self.assertIn('fresh output directory',completed.stderr)

if __name__ == '__main__': unittest.main()
