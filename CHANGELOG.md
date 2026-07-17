# Changelog

## 0.1.0

- Initial release.
- SIMD dot product, cosine similarity, Euclidean distance, and
  normalization over `Float32List`, with fail-fast validation of
  lengths and non-finite components.
- `VectorMatrix`: packed row-major storage with precomputed row norms,
  top-k cosine, dot product, and Euclidean search, and a `VKT1` binary
  format for serialization.
