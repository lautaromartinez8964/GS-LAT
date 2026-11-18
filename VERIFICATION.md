# Implementation verification

In the project environment, the standalone prediction code was replayed against the eight stored masks from the selected empirical two-class, support-entropy, unfloored arm. Valid domains, seed fractions, accepted source counts and final masks matched in Assam, Poyang Lake, Zhengzhou Southwest, Mozambique AOI02, Da Nang, the dense and sparse windows of S1S2 Scene 7, and the Scene 19 window. All eight masks had zero differing pixels. Predictions were compared with stored algorithm masks without opening reference labels or modifying archived products.

This is an implementation-equivalence check on development cases, not an independent accuracy evaluation. The checked dependency versions are in `requirements-verified.txt`. The scene data and comparison masks are not included.
