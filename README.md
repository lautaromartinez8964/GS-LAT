# GS-LAT

**Hybrid Geometric-Statistical Local Adaptive Thresholding for SAR Images: Water Segmentation as a Test Case**

GS-LAT segments dark water candidates from a single-band SAR raster. Ordered boundary profiles provide both local geometric thresholds and two scene-wide empirical class histograms. The geometric support strength `q` combines the geometric threshold with a local statistical threshold. This package implements only the empirical two-class version without a dark-mean threshold floor or an entropy fusion factor.

The same workflow processes supplied linear power, amplitude DN or dB values. It does not take logarithms of SAR pixel values. Changing the radiometric representation can change the result. Prediction needs no reference labels and does not identify flood change between dates.

## Install and run

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install .
gs-lat --input sar.tif --output-dir run_power --representation positive
gs-lat --input sar_db.tif --output-dir run_db --representation db
```

`python -m gslat.cli` is equivalent. `positive` requires strictly positive valid linear power or amplitude values; `db` accepts supplied finite dB values of either sign. The representation flag is explicit and is not inferred from observed values. Mark invalid pixels with NoData or a raster mask. Decode stored scale factors such as centi-dB before running. Use a georeferenced raster with a meaningful pixel size, preferably projected with square pixels. The full image is processed in memory, and running on a crop changes scene statistics and geometry. Use a fresh output directory. If the geometry or empirical class gates yield no valid source, the command fails explicitly.

`mask.tif` uses 1 for dark/water candidates, 0 for other valid pixels, and 255 for NoData. Diagnostic rasters include the seed, core, edges, support, support strength, geometry weight, anchor, local prior, statistical threshold and final threshold. Float rasters use NaN for NoData; the decision is made before the float32 export. `run.json` records the input file hash and run diagnostics. Preserve external mask sidecars with the input if present.

See [METHOD.md](METHOD.md) for the processing rules, [PROVENANCE.md](PROVENANCE.md) for source history and [VERIFICATION.md](VERIFICATION.md) for implementation checks. This research implementation has known recall limitations. Images, references, predictions and manuscript files are not included.
