# vector_kit

Exact top-k cosine, dot, and Euclidean search over embeddings held in one
packed buffer, plus pairwise `dot`, `cosineSimilarity`, and
`euclideanDistance` on `Float32List`. Every query reads every row; this is
not an approximate index.

Do not import this package for a few hundred vectors. A hand-written loop
is the right answer until a query costs milliseconds. The measurement is
`bench/break_even.dart`: 10 to 100,000 rows × 768 dims, k=10, Dart VM, the
same scan-and-sort `bench/bench.dart` uses. A cosine loop plus a sort is
5.9 µs at N=10 against 1.9 µs for `VectorMatrix.topKCosine`, 699 µs against
136 µs at N=1,000, 2.28 ms against 429 µs at N=3,200, and 86.3 ms against
13.4 ms at N=100,000. The package is faster at every measured N; it still
does not pay for a dependency below a few thousand rows, where the loop is
under a millisecond. dart2js and dart2wasm were not part of that sweep
(it does not launch a browser). The web comparison is still 1000 rows ×
384 dims (`test/platform_cost_test.dart`, k=10): a packed scalar scan with
cached row norms costs 257 µs/query on the Dart VM, 258 µs on dart2js,
272 µs on dart2wasm. `VectorMatrix.topKCosine` is 77 µs on the VM (3.3×
faster than that loop) and slower than the loop on the web (322 µs dart2js,
292 µs dart2wasm). All of those are well under a millisecond.

## Usage

From `example/vector_kit_example.dart`:

```dart
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

void main() {
  final a = Float32List.fromList([1, 0, 2, 1]);
  final b = Float32List.fromList([2, 1, 1, 0]);
  print(dot(a, b)); // 4.0

  final index = VectorMatrix.fromRows([
    [1.0, 0.0, 0.0, 0.0],
    [0.7, 0.7, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
  ]);
  final query = Float32List.fromList([0.9, 0.1, 0.0, 0.0]);
  for (final (doc, score) in index.topKCosine(query, 2)) {
    print('doc $doc scores ${score.toStringAsFixed(4)}');
  }
}
```

`VectorMatrix.add` copies a `Float32List` row. `topKCosine`, `topKDot`, and
`topKEuclidean` take any `List<double>`; a model embedding does not need
wrapping.

## Contracts

Layout. `VectorMatrix` packs rows back to back in one `Float32List`, each
padded to a multiple of four components so the next row starts on a 16-byte
boundary. Padding is internal: `rowAt`, `toBytes`, and `dimension` expose
the logical row only.

Cached at insert. `VectorMatrix.add` stores the L2 norm. `topKCosine` uses
that cache (one dot per row). `rowAt` returns a live view, not a copy;
writing through it updates storage but not the cached norm, so cosine
scores for that row go stale. There is no row-update method.
`VectorMatrix.fromBytes` rebuilds norms from the payload. Layout: ASCII
`VKT1`, little-endian uint32 dimension and count, then row-major float32.
Corrupt input throws `FormatException`.

Normalisation. Rows need not be unit length. `cosineSimilarity` and
`VectorMatrix.topKCosine` divide by both L2 norms and clamp to `[-1, 1]`.
`normalizeInPlace` and `normalized` reject a zero vector (`ArgumentError`).
Pairwise functions accept `Float32List` only, not `List<double>`.

Quantized cost. `QuantizedMatrix.from` stores int8 components plus per-row
scale and the L2 norm of the dequantized stored row, not the source.
`bench/quantization_benchmark.dart` on 5,000 × 768 unit vectors: 14.6 MB
→ 3.7 MB (3.9× smaller), recall@10 99.3% of the float top-10, search 664 →
2723 µs/query (4.1× slower). That recall is an upper bound for random
vectors; clustered embeddings confuse eight bits. `from` does not mutate
the source. Byte rows cannot take the SIMD float path; this buys memory,
not speed. `QuantizedMatrix.topKCosine` does not clamp; range holds because
the cached norm is of what is stored.

Methods.

- Free: `dot`, `cosineSimilarity`, `euclideanDistance`, `normalizeInPlace`,
  `normalized`.
- `VectorMatrix`: `VectorMatrix(dimension)`, `fromRows`, `fromBytes`, `add`,
  `rowAt`, `topKCosine`, `topKDot`, `topKEuclidean`, `toBytes`, `rowCount`,
  `dimension`. No `byteSize`, no update, no remove.
- `QuantizedMatrix`: `from`, `topKCosine`, `topKDot`, `topKEuclidean`,
  `rowCount`, `dimension`, `byteSize`. No `add`, no `rowAt`, no `toBytes` /
  `fromBytes`.

Scoring. Cosine and dot: larger is better. Euclidean: the score is the
distance, smaller is better. `topKCosine` skips zero-norm rows and rejects
a zero query. `topKDot` and `topKEuclidean` allow a zero query. `k < 1`
throws `ArgumentError`. At most `rowCount` hits; cosine can return fewer
than `k`. Equal-score order is unspecified.

Precision. VM kernels accumulate in float32; off-VM scalar kernels in
double. Scores differ by about 5e-9; ranking does not
(`test/cross_platform_test.dart`). Pairwise `dot` stays within 1e-5
relative to the product of input norms up to dimension 1024.

## Mistakes

- N=500. Symptom: a dependency for a sub-millisecond loop. Fix: write the
  scan; `bench/break_even.dart` is the sweep.
- Write through `rowAt`. Symptom: wrong `topKCosine` ranks. Fix: treat the
  view as read-only; rebuild if rows change.
- `dot` on `List<double>`. Symptom: compile error. Fix: `Float32List.fromList`.
- `add` / `rowAt` / `toBytes` on `QuantizedMatrix`. Symptom: compile error.
  Fix: fill a `VectorMatrix`, then `QuantizedMatrix.from`.
- `QuantizedMatrix` for speed. Symptom: ~4× slower queries. Fix: use it
  only when the float matrix does not fit.
- Sort Euclidean scores descending. Symptom: farthest rows first. Fix:
  smaller score is nearer.
- Zero vector into `cosineSimilarity` or `topKCosine`. Symptom:
  `ArgumentError`.
- Off the Dart VM. `Float32x4` is real SIMD only on the VM. Elsewhere the
  SDK emulates it (dart2js: four boxed doubles per lane; dart2wasm:
  `NaiveFloat32x4`), which is slower than a scalar loop. Through 1.0.4,
  1000×384 `topKCosine` was 4,780 µs (dart2js) and 12,102 µs (dart2wasm)
  against 258 / 272 µs for the hand-written loop (18.5× and 44.5× slower).
  Since 1.1.0, `simd.dart` compiles SIMD on the VM and scalar kernels when
  `dart.library.js_interop` is present. Do not put `Float32x4` back on the
  web path. Do not expect bit-identical scores across VM and web.
- Editing only `simd_native.dart` or only `simd_web.dart`. Symptom: the
  other target breaks. Fix: keep names and signatures identical.
- Treating this as an approximate index. Symptom: time grows linearly with
  `rowCount`. There is no build step and no filter DSL.
- Putting `VectorKitStore` in `lib/` so it can implement rag_kit's
  `VectorStore`. Symptom: rag_kit becomes a runtime dependency of every
  consumer. The adapter stays in `example/vector_kit_store.dart`; rag_kit
  is a dev dependency. Copy the file into an app that depends on both.
- `VectorKitStore` for a few hundred rag_kit chunks, or on the web, for
  speed. Symptom: a dependency for a sub-millisecond loop, or a slowdown
  on dart2js / dart2wasm. `InMemoryVectorStore` already caches norms and
  uses a k-heap. The packed scan pays off on the Dart VM from a few
  thousand chunks up.

## Where

- `package:vector_kit/vector_kit.dart` — public export.
- `src/ops.dart` — pairwise functions.
- `src/vector_matrix.dart` — `VectorMatrix`, `QuantizedMatrix`.
- `simd.dart` — kernel selection; do not import `simd_native.dart` or
  `simd_web.dart` directly.
- `example/` — `vector_kit_example.dart`, `semantic_search.dart` (20,000 ×
  384: 29.3 MB float32, 7.6 MB int8, recall@10 100% on that random data),
  `vector_kit_store.dart` (a rag_kit `VectorStore`; not in `lib/`),
  `with_rag_kit.dart`.
- Tests: `dart test`. Ranking across VM / dart2js / dart2wasm:
  `test/cross_platform_test.dart`. rag_kit `VectorStore` contract:
  `test/rag_kit_store_test.dart`.
- Platform timings (print only):
  `dart test test/platform_cost_test.dart -t bench`, and the same with
  `-p chrome` and `-p chrome -c dart2wasm`.
- `dart run bench/bench.dart`: 768-d dot and 10k/100k top-k.
- `dart run bench/break_even.dart`: N sweep of top-k vs loop+sort, Dart VM.
- `dart run bench/quantization_benchmark.dart`: int8 figures.
- `doc/web-performance.md`: platform table.

SDK `^3.8.0`. `dart analyze` must stay clean.
