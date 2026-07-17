import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

void main() {
  // Pairwise operations on Float32List.
  final a = Float32List.fromList([1, 0, 2, 1]);
  final b = Float32List.fromList([2, 1, 1, 0]);
  print('dot:      ${dot(a, b)}');
  print('cosine:   ${cosineSimilarity(a, b).toStringAsFixed(4)}');
  print('distance: ${euclideanDistance(a, b).toStringAsFixed(4)}');

  final unit = normalized(a);
  print('normalized a: ${unit.map((x) => x.toStringAsFixed(4)).join(', ')}');

  // Top-k search over a packed matrix. In practice the rows are
  // embeddings from your model; the toy vectors keep the output
  // readable.
  final index = VectorMatrix.fromRows([
    [1.0, 0.0, 0.0, 0.0], // doc 0
    [0.7, 0.7, 0.0, 0.0], // doc 1
    [0.0, 1.0, 0.0, 0.0], // doc 2
    [0.0, 0.0, 1.0, 0.0], // doc 3
  ]);
  final query = Float32List.fromList([0.9, 0.1, 0.0, 0.0]);
  for (final (doc, score) in index.topKCosine(query, 2)) {
    print('doc $doc scores ${score.toStringAsFixed(4)}');
  }

  // The matrix serializes to a compact binary form and restores with
  // its precomputed norms rebuilt.
  final restored = VectorMatrix.fromBytes(index.toBytes());
  print('restored ${restored.rowCount} rows of ${restored.dimension} dims');
}
