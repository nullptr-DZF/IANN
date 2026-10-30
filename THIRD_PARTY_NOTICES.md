# Third-party software

The modified headers in `hnswlib/` derive from [hnswlib](https://github.com/nmslib/hnswlib), distributed under the Apache License, Version 2.0. The upstream license is retained in `THIRD_PARTY_LICENSES/hnswlib.LICENSE`.

The CPU feature-detection implementation in `hnswlib/hnswlib.h` credits [FeatureDetector](https://github.com/Mysticial/FeatureDetector).

Modifications add experimental search instrumentation and preserve large-index I/O and CPU distance-kernel selection.
