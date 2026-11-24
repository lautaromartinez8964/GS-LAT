# Code provenance

This package originally ported the `empirical_support_entropy_unfloored` arm from the November 18, 2026 development experiment. On November 19, 2026 the author fixed the fusion weight to the geometric support `q`, removing the binary-entropy factor. The native histogram initializer and empirical class statistics were extracted from the research experiment drivers. Shared boundary-profile geometry and binary cleanup build on the project's previously released BP-LAT geometry package. The original MIT license and contributor notice are retained.

On November 24, 2026 the author removed the halfway shrinkage toward the scene-wide initial prior. The statistical branch still computes the posterior once and uses the clipped neighborhood average as the local prior. The geometric fusion rule is unchanged.
