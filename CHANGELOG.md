## 0.4.0

- Mark `VectorMatrix` and `QuantizedMatrix` as `final`, ahead of a 1.0.0
  freeze. Neither was designed to be subtyped: they are concrete data
  structures, cheap to construct, and nothing in the package, its tests, its
  examples or its benchmarks extends or implements either. Sealing them keeps
  the rest of 1.x additive, because the planned work (the `QuantizedMatrix`
  parity gaps, an ANN index) adds members to exactly these types, and every
  addition would otherwise break anyone who had implemented them. Adding
  `final` after 1.0.0 would require a major version; removing it later would
  not, so this is the direction that stays open. No behaviour change.

## 0.3.1

- `QuantizedMatrix.topKCosine` and `topKDot` now reject a query with a NaN or
  infinite component instead of returning a result list whose scores are all
  NaN. `VectorMatrix` already validated this; `QuantizedMatrix` only checked
  `k` and the query length, so the same bad query that throws on one matrix
  type silently poisoned the ranking on the other.

## 0.3.0

- The matrix search methods take a plain `List<double>` query, not only a
  `Float32List`. An embedding straight from a model is a `List<double>`, so
  building the index with `fromRows` and then calling `topKCosine` used to
  compile on the first line and fail on the second, which is exactly the kind of
  seam a caller trips on. The widening is source-compatible: a `Float32List` is
  a `List<double>`, so existing calls are unchanged. Applies to `topKCosine`,
  `topKDot` and `topKEuclidean` on both `VectorMatrix` and `QuantizedMatrix`.
- `example/semantic_search.dart` is the real use case: a 20,000-document index
  of 384-dim vectors, searched, with the result measured. A top-5 query is
  1.4 ms against 15.9 ms for the hand-written cosine loop (11x), and the int8
  `QuantizedMatrix` holds the index in a quarter of the memory, with the recall
  cost measured against the float ranking rather than assumed.

## 0.2.2

- Shorten the screenshot description. pub.dev accepts up to 200 characters but
  scores only those under 160, so the previous release published cleanly and
  quietly gave up the documentation points it was meant to earn.

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
