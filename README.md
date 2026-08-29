# vector_kit

A top-10 search over 100,000 embeddings costs 82 ms per query when you score
every row with a scalar cosine loop and sort the results. The same search here
costs 13.3 ms: one SIMD dot product per row of a single packed buffer, row
norms cached at insert time, and a bounded heap in place of the sort. Both
figures come from one command on an Apple M-series laptop, 768 dimensions:

```
dart run bench/bench.dart
```

![Benchmark chart. Dot product over 768 dimensions, nanoseconds per call: a scalar loop over a list of doubles takes 665 ns, a Float32List loop 492 ns, SIMD with a single accumulator about 153 ns, and vector_kit 142 ns, which is 4.7 times faster than the list-of-doubles loop. Top-10 cosine over 10,000 rows: full scan and sort 7.2 ms per query against 1.4 ms, 5.3 times faster. Over 100,000 rows: 82 ms against 13.3 ms, 6.2 times faster.](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/bench.png)

![The semantic search example running: a set of sentences is embedded, a query
is matched against them, and the nearest ones come back with their scores](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/demo.gif)

## When not to import it

Do not take this dependency for a few hundred vectors. A hand-written cosine
loop plus a sort is 5.9 µs at 10 rows of 768 dimensions.
`VectorMatrix.topKCosine` is faster at every size we measured — 1.9 µs at
those same 10 rows — and that is still not a reason to import a package.
Write the loop until a query costs milliseconds.

```
dart run bench/break_even.dart
```

Dart VM, JIT, 768-d, k=10. The baseline is the same scan-and-sort
`bench/bench.dart` uses at 10,000 and 100,000 rows:

|    rows | loop + sort | topKCosine | vs the loop |
| ------: | ----------: | ---------: | ----------: |
|      10 |      5.9 µs |     1.9 µs |        3.1x |
|      32 |     20.2 µs |     5.1 µs |        3.9x |
|     100 |     63.8 µs |    14.7 µs |        4.3x |
|     320 |    215.9 µs |    45.5 µs |        4.7x |
|   1,000 |    699.3 µs |   136.2 µs |        5.1x |
|   3,200 |     2.28 ms |   429.1 µs |        5.3x |
|  10,000 |     7.44 ms |    1.35 ms |        5.5x |
|  32,000 |    26.33 ms |    4.38 ms |        6.0x |
| 100,000 |    86.29 ms |   13.41 ms |        6.4x |

The loop first costs a millisecond between 1,000 rows (699 µs) and 3,200
(2.28 ms). That is the size where the packed scan starts to pay for itself.
Below it, skip the package.

This sweep is the Dart VM. dart2js and dart2wasm were not measured here:
they need Chrome, and this run did not launch a browser. The one web
comparison in the repo is 1,000 × 384 in `test/platform_cost_test.dart`,
under [Off the Dart VM](#off-the-dart-vm).

## Why this instead of what you already have

**Instead of a scalar loop.** The chart above comes from `dart run
bench/bench.dart`, which is in the repo and measures both sides in one process.
A run on an Apple M-series laptop gives 647 ns per call for the `List<double>`
dot product against 143 ns for `dot`, and 84.4 ms per query for a full scan and
sort over 100,000 x 768 against 13.9 ms for `VectorMatrix.topKCosine`
(`lib/src/vector_matrix.dart:236`). Those figures move a few percent run to run.
The harness also checks that both sides return the same ten rows in the same
order, and prints the agreement count next to the timings.

**Instead of `ml_linalg`.** It is the established SIMD linear algebra package
for Dart and it is good at what it covers, including `getCosine` and
`distanceTo(..., distance: Distance.cosine)` between two vectors
(`lib/vector.dart:352` and `:346`). Search its 6,544 lines and its README for
`topk`, `top_k`, `nearest`, or `knn` and there are no hits. Searching a matrix of
embeddings still means your own loop over every row and a sort at the end, which
is the 84.4 ms above.

**Reach for it when**

- You hold more than a few thousand embeddings in memory and a query has to feel
  instant.
- You are doing semantic search on-device, where a hosted vector index is not an
  option.
- You need the top k rather than one pairwise distance, and want the same answer
  a naive scan would give.

Skip it if you have a few hundred vectors. The table above is that
measurement: a plain loop finishes in tens to hundreds of microseconds, and
the packed-buffer bookkeeping is cost with nothing behind it.

Dart has shipped SIMD types in `dart:typed_data` for years. Using `Float32x4`
well means alignment rules, scalar tails for lengths that are not a multiple of
four, accumulator ordering, and one platform trap that has its own section
below. vector_kit is that wiring as a pure Dart package with no runtime
dependencies.

## Quick start

```dart
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

void main() {
  final a = Float32List.fromList([1, 0, 2, 1]);
  final b = Float32List.fromList([2, 1, 1, 0]);
  print(dot(a, b)); // 4.0
  print(cosineSimilarity(a, b)); // 0.666...
  print(euclideanDistance(a, b)); // 2.0

  // Top-k search: pack the corpus once, query many times.
  final index = VectorMatrix(768);
  for (final embedding in embeddings) {
    index.add(embedding); // Float32List of length 768
  }
  for (final (row, score) in index.topKCosine(queryEmbedding, 10)) {
    print('row $row scores $score');
  }
}
```

The query is any `List<double>`. An embedding straight out of a model goes into
`topKCosine`, `topKDot` or `topKEuclidean` without being wrapped in a
`Float32List` first.

`example/vector_kit_example.dart` is a short API tour.
`example/semantic_search.dart` is the real job: it builds a 20,000-document
index of 384-dimension vectors, searches it, and reports what it cost. That
index takes 29.3 MB as float32 and 7.6 MB once quantized to int8, with a
recall@10 of 100% on the demo's data.

## Measured performance

From `bench/bench.dart` on an Apple M-series laptop, Dart 3.11, JIT,
768-dimensional vectors:

| Workload                   | Baseline                          | vector_kit    | Speedup |
| -------------------------- | --------------------------------- | ------------- | ------- |
| dot, 1M calls              | 665 ns/call, list-of-doubles loop | 142 ns/call   | 4.7x    |
| dot, 1M calls              | 492 ns/call, `Float32List` loop   | 142 ns/call   | 3.5x    |
| topKCosine k=10, 10k rows  | 7.2 ms/query, full scan and sort  | 1.4 ms/query  | 5.3x    |
| topKCosine k=10, 100k rows | 82.0 ms/query, full scan and sort | 13.3 ms/query | 6.2x    |

Compiled ahead of time (`dart compile exe`) the same benchmark gives
126 ns/call for dot and 1.0 / 10.5 ms/query for the two top-k workloads. The
four independent accumulators in the dot kernel are worth about 8 percent over
a single accumulator under the JIT and 14 percent compiled; the benchmark
measures both variants.

The baseline in that table scores every row and then sorts every score. A
carefully hand-written scan is a harder yardstick: cache the row norms, keep a
k-sized insertion list, and the gap narrows to 3.3x at 1000 rows of 384
dimensions. `test/platform_cost_test.dart` runs that version too, on all three
targets, and the numbers are in
[doc/web-performance.md](https://github.com/Yusufihsangorgel/vector_kit/blob/main/doc/web-performance.md).
Run both on your own hardware before relying on either.

Where the top-k gain comes from: one SIMD dot product per row over a single
packed buffer, L2 norms precomputed when rows are added, and a bounded min-heap
instead of sorting all scores.

## What is inside

Functions over `Float32List`:

| Function              | Notes                                         |
| --------------------- | --------------------------------------------- |
| `dot(a, b)`           | four `Float32x4` accumulators, scalar tail    |
| `cosineSimilarity`    | clamped to `[-1, 1]`, rejects zero vectors    |
| `euclideanDistance`   | L2 distance                                   |
| `normalizeInPlace(v)` | scales `v` to unit norm, rejects zero vectors |
| `normalized(v)`       | same, but returns a copy                      |

`VectorMatrix` stores rows back to back in one `Float32List`, padded to a
multiple of four components so every row starts on a 16-byte boundary. The
search loops read the whole matrix through a single `Float32x4List` view: no
per-row alignment checks, no tails.

![vector_kit memory layout: rows packed end to end in one Float32List with padding, viewed through a single Float32x4List for top-k search](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/layout.png)

- `add(row)` copies the row in and caches its L2 norm.
- `topKCosine(query, k)`, `topKDot(query, k)`, and `topKEuclidean(query, k)`
  return `(index, score)` records, best first. For Euclidean the score is the
  distance itself: smaller is better.
- `rowAt(index)` returns a live view into the storage, not a copy. Treat it as
  read-only; writing through it leaves the cached norm stale.
- `toBytes()` and `VectorMatrix.fromBytes(bytes)` serialize to a simple binary
  format: the ASCII magic `VKT1`, dimension and row count as little-endian
  uint32, then the float32 components in row-major order. Corrupt input throws
  `FormatException`.

## Off the Dart VM

`Float32x4` is a real SIMD type only on the Dart VM. Everywhere else the SDK
emulates it: dart2js backs it with four boxed doubles and allocates a fresh
object on every lane read, and dart2wasm's own patch file names its version
`NaiveFloat32x4`. Emulated SIMD runs slower than writing no SIMD at all, which
is a trap this package fell into and shipped through 1.0.4.

```
dart test test/platform_cost_test.dart -t bench
dart test test/platform_cost_test.dart -t bench -p chrome
dart test test/platform_cost_test.dart -t bench -p chrome -c dart2wasm
```

![The cost of one top-k cosine search on three targets, each measured against the same search hand-written as a plain scalar loop on that same target. With Float32x4 kernels compiled everywhere: 0.31 times the loop on the native VM, 18.5 times the loop on dart2js, 44.5 times on dart2wasm. With SIMD on the VM and scalar kernels elsewhere, which is what 1.1.0 ships: 0.30 times on the VM, 1.25 times on dart2js, 1.07 times on dart2wasm.](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/platform.png)

**If you ship to web on 1.0.4 or earlier, upgrade.** Since 1.1.0 the kernel set
is chosen at compile time from `dart.library.js_interop`: SIMD on the VM, plain
scalar loops elsewhere. Nothing in the public API changed. On the web the
package now costs about what the hand-written loop costs, and the packed
storage, the persistence and the int8 path all come with it.

The scalar kernels accumulate in double where the VM kernels accumulate in
float32, which puts web scores about 5e-9 away from VM scores and makes them
marginally more accurate. Ranking is asserted rather than assumed:
`test/cross_platform_test.dart` pins the exact top-10 rows for cosine, dot and
euclidean, and CI runs it on the VM, on dart2js and on dart2wasm.

## Validation

Every operation fails fast instead of letting a bad component poison scores
downstream: length mismatches, empty vectors, NaN or infinite components, and
zero vectors where the operation is undefined all throw `ArgumentError` at the
call site.

The finiteness check costs nothing on the hot path. A NaN or infinite component
always drives a multiply-add accumulation non-finite, and infinities never
cancel back to a finite value. Only a non-finite result triggers a rescan of
the inputs to locate the exact offending component.

## Precision

Accumulation happens in `Float32x4` lanes rather than in double, which puts
results a little away from an exact double-precision sum. The test suite pins
the difference to within 1e-5 relative to the product of the input norms at
dimensions up to 1024. The scalar tail (the last `length % 4` components)
accumulates in double precision. Components tiny enough to underflow float32
(below about 1e-38) can therefore contribute or vanish depending on their
position; real embedding values sit many orders of magnitude above that floor.
If you need double-precision accumulation, this package is the wrong tool.

Inputs are accepted as any `Float32List`, including views. A view that does not
start on a 16-byte boundary is copied internally before the SIMD loop, and
`normalizeInPlace` still writes the result back to the original view.

## int8 quantization

When the vectors stop fitting comfortably in memory, `QuantizedMatrix` stores
each row as one byte per dimension plus the scale that undoes it:

```dart
final matrix = VectorMatrix.fromRows(embeddings);
final compact = QuantizedMatrix.from(matrix);

final hits = compact.topKCosine(query, 10);
```

![int8 quantization on 5,000 vectors of 768 dimensions: memory drops from 14.6 MB to 3.7 MB, 3.9 times smaller; 99.3% of the float top-10 survives quantization; search is 4.1 times slower, 664 to 2723 microseconds per query, because the int8 rows cannot take the SIMD float path, which buys memory and costs throughput.](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/quantization.png)

`benchmark/quantization_benchmark.dart`, seeded, on an Apple M-series core:

| | float32 | int8 |
|---|---|---|
| memory | 14.6 MB | 3.7 MB (3.9x smaller) |
| search | 664 µs/query | 2723 µs/query (4.1x the time) |
| recall@10 | n/a | 99.3% of the float top-10 |

This buys memory and costs throughput: the byte rows cannot go through the same
SIMD path the float rows do. Reach for it when the corpus is the problem rather
than the latency.

Take that recall as an upper bound rather than a promise. Those are uniformly
random vectors, which sit far apart in 768 dimensions, and rounding rarely
reorders them; packing more vectors into the same space lowers it. Real
embeddings cluster, and clustered neighbours are exactly the ones eight bits
confuse. `QuantizedMatrix.from` leaves the source matrix untouched precisely so
you can measure recall on your own vectors before trusting it.

## What this is not

This is not a database and not an index. Every query reads every row, which is
what the 13.3 ms at 100,000 x 768 buys and also what it costs: ten times the
corpus is ten times the work, with no build step to amortize it against. Past
the size where that hurts you want an approximate index, and on device that
means [objectbox](https://pub.dev/packages/objectbox), which keeps vectors in
an HNSW index next to your other fields and queries them together.

What the full scan buys back is that the answer is the true top-k, with no
tuning parameter between you and it. An approximate index filtered after the
fact can return fewer rows than you asked for, because the filter runs over the
neighbours the index already picked rather than over the corpus; ObjectBox
documents that on `maxResultCount`, and the request to spell it out in the
vector-search guide has been
[open since 2024](https://github.com/objectbox/objectbox-dart/issues/658).
Here the equivalent move is exact: `topKCosine(query, matrix.rowCount)` scores
and orders every row, and filtering that list afterwards still leaves you the
true top-k of whatever survives.

Missing on purpose: no metadata or filter DSL, since `topK*` returns row
indices and what a row means is yours to store; no isolate pool; no
`Float64List` path; no persistence past `toBytes()`.

## Relation to rag_kit

[rag_kit](https://pub.dev/packages/rag_kit) is the retrieval pipeline
(chunking, embedding, context building) and this package is a numeric
backend for it. rag_kit ships `InMemoryVectorStore`, which scores every
row with a scalar cosine loop, caches the L2 norm, and keeps a k-heap.
That is already the careful scan, not the naive sort in the break-even
table above.

`example/vector_kit_store.dart` is a `VectorStore` that uses
`VectorMatrix.topKCosine` on the unfiltered path and scores only the
rows that pass `where`. It is **not** exported from
`package:vector_kit/vector_kit.dart`. Implementing rag_kit's interface
requires importing rag_kit, and putting that import in `lib/` would make
rag_kit a runtime dependency of every vector_kit user. rag_kit is a dev
dependency here. Copy the adapter into an app that already depends on
both packages:

```dart
import 'package:rag_kit/rag_kit.dart';

import 'vector_kit_store.dart'; // the copied file

final retriever = Retriever(
  embedder: myEmbedder,
  store: VectorKitStore(),
  chunker: Chunker.paragraphs(),
);
```

**When it is worth it.** On the Dart VM, a query over the careful loop
first costs a millisecond between 1,000 and 3,200 rows of 768
dimensions (`dart run bench/break_even.dart`). Below a few thousand
chunks — a handbook, a small notes corpus — keep
`InMemoryVectorStore`. From a few thousand up, and clearly at the
10k–100k sizes rag_kit names as its range, the packed scan is the
better backend on the VM (5.5x at 10,000 rows, 6.4x at 100,000 against
the sort; about 3x against the careful loop at 1,000 × 384). On the
web, `VectorMatrix.topKCosine` is slower than that loop (322 µs vs
258 µs dart2js, 292 µs vs 272 µs dart2wasm at 1,000 × 384), so this
adapter is not a speedup there.

The store keeps the documents plus a packed copy of the embeddings, so
the float32 footprint is roughly twice `InMemoryVectorStore` at the same
count. Replacing or removing a row rebuilds the matrix; index once and
query many times.

`test/rag_kit_store_test.dart` sends the same queries through both
stores and asserts the same document order, including ties.

## Planned

Approximate nearest neighbour search (HNSW) and isolate-parallel search for
very large matrices. Both stay out until the exact-search core has settled.

## License

MIT.
