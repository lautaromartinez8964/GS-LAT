# Implementation verification

# Implementation verification

Version 0.1.0 replayed the empirical two-class, support-entropy, unfloored arm against eight stored development masks and matched all eight pixel for pixel. That check applied only to the entropy-weighted formula and is not a verification of the current package.

Version 0.2.0 sets the fusion weight to the geometric support `q` and removes the entropy factor. It is therefore expected to differ from those stored masks. The retained automated check is the unit test for the empirical risk lookup and connected-component cleanup. This package has not been independently accuracy-validated. The checked dependency versions are in `requirements-verified.txt`. Scene data and comparison masks are not included.
