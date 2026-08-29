# vector_kit example

Three programs. `vector_kit_example.dart` shows the whole public surface on toy
vectors small enough that every number stays readable. `semantic_search.dart`
runs the job the package exists for, at a size where the layout starts to
matter. `with_rag_kit.dart` wires `VectorKitStore` into rag_kit's `Retriever`.

## The API tour

```dart
// Pairwise operations on Float32List.
print(dot(a, b));                 // 4.0
print(cosineSimilarity(a, b));    // 0.6667
print(euclideanDistance(a, b));   // 2.0000
final unit = normalized(a);       // 0.4082, 0.0000, 0.8165, 0.4082

// Top-k search over a packed matrix. In practice the rows are your model's
// embeddings; these toy rows keep the scores legible.
final index = VectorMatrix.fromRows(rows);
for (final (doc, score) in index.topKCosine(query, 2)) {
  print('doc $doc scores $score');
}

// Serialize to a compact binary form and back, norms rebuilt on restore.
final restored = VectorMatrix.fromBytes(index.toBytes());
```

`VectorMatrix.fromRows` is doing more than holding a list of lists. It copies
every row into one `Float32List`, pads each to a multiple of four components so
the next row still starts on a 16-byte boundary, and caches the row's L2 norm
while it has the data in hand. That is what lets the search read the whole
matrix through a single `Float32x4List` view, and it is why cosine costs one
dot product per row instead of three passes:

![vector_kit memory layout: rows packed end to end in one Float32List with padding, viewed through a single Float32x4List for top-k search](https://raw.githubusercontent.com/Yusufihsangorgel/vector_kit/main/doc/layout.png)

Run it:

```
dart run example/vector_kit_example.dart
```

Output:

```
dot:      4.0
cosine:   0.6667
distance: 2.0000
normalized a: 0.4082, 0.0000, 0.8165, 0.4082
doc 0 scores 0.9939
doc 1 scores 0.7809
restored 4 rows of 4 dims
```

The round trip at the end is worth a look if you plan to cache an index on
disk. `toBytes()` writes the ASCII magic `VKT1`, the dimension and row count as
little-endian uint32, then the components in row-major order. `fromBytes`
validates all of that and recomputes the norms rather than trusting them, so a
truncated or corrupt file throws `FormatException` instead of quietly returning
wrong neighbours.

## The realistic one

`semantic_search.dart` builds 20,000 documents with metadata and deterministic
fake embeddings of 384 dimensions, runs a cosine query against the lot, and
then measures two things worth knowing before you take the dependency: how
`VectorMatrix` compares with the nested-list loop most people write first, and
what the int8 `QuantizedMatrix` costs in recall for what it saves in memory.

```
dart run example/semantic_search.dart
```

The memory figures are deterministic because the embeddings are seeded: the
same index takes 29.3 MB as float32 and 7.6 MB as int8, and on this data every
one of the float top-10 survives quantization. The timings it prints are your
machine's, and they move with what else is running on it. The package README
has the controlled numbers, taken from `bench/bench.dart`, along with the
recall caveat that matters here: uniformly random vectors sit far apart in high
dimensions, and real embeddings cluster, which is exactly the case eight bits
find hardest.

## rag_kit

`vector_kit_store.dart` is a rag_kit `VectorStore` over `VectorMatrix`. It is
not in `lib/`: the interface types live in rag_kit, and importing them from
the public library would make rag_kit a runtime dependency of every
vector_kit user. Copy that file into an app that already depends on both
packages.

It is the better backend on the Dart VM from a few thousand chunks up, which
is where a cosine loop first costs a millisecond. Below that, and on the web,
keep rag_kit's `InMemoryVectorStore`. The package README spells out the
sizes. `test/rag_kit_store_test.dart` asserts that the same query through both
stores returns the same document order, including ties.

```
dart run example/with_rag_kit.dart
```
