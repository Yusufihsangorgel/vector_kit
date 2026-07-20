## 0.2.1

- Declare the benchmark chart in `pubspec.yaml` so pub.dev renders it on the
  package page. The chart was already in the repository and the README, but
  pub.dev shows only what the `screenshots:` field points at, so the page a
  reader lands on from search opened with text where the measurement should
  have been.

## 0.2.0

- Add `QuantizedMatrix`, an int8 form of `VectorMatrix` for corpora that no
  longer fit comfortably in memory. Each row is scaled so its largest component
  maps to 127 and stored as one byte per dimension. Measured on 5,000 rows of
  768 dimensions: 14.6 MB becomes 3.7 MB, 3.92x smaller, while a search goes
  from 0.63 ms to 2.50 ms a query, 3.96x the time, because the byte rows cannot
  use the SIMD path the float rows do. It buys memory and costs throughput,
  which is the trade to make only when the corpus is the problem.
- `QuantizedMatrix.from` leaves the source matrix usable, so recall can be
  measured against the exact ranking on real vectors. Recall@10 was 100% on the
  benchmark corpus, but that is an upper bound: uniformly random vectors sit
  far apart in high dimensions, and real embeddings cluster, which is where
  eight bits start confusing neighbours.

## 0.1.1

- Docs: tightened the README wording and visuals.

# Changelog

## 0.1.0

- Initial release.
- SIMD dot product, cosine similarity, Euclidean distance, and
  normalization over `Float32List`, with fail-fast validation of
  lengths and non-finite components.
- `VectorMatrix`: packed row-major storage with precomputed row norms,
  top-k cosine, dot product, and Euclidean search, and a `VKT1` binary
  format for serialization.
