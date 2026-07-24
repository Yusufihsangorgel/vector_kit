# vector_kit example

`vector_kit_example.dart` shows the core surface on toy vectors kept small enough
that the output is readable: the pairwise operations, a top-k cosine search over
a packed matrix, and the matrix's compact binary round-trip.

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

`semantic_search.dart` is the closer-to-real version: documents with metadata
and deterministic fake embeddings, a query that finds the nearest by cosine,
then `VectorMatrix` measured against a plain nested-list loop at a realistic
size and the int8 `QuantizedMatrix` trading a little recall for a quarter of the
memory. The memory-vs-recall numbers are charted in the package README.
