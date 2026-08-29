// The baseline a caller writes instead of taking this package: score every
// row with a scalar cosine, then sort. Shared by bench.dart (the published
// 10k / 100k numbers) and break_even.dart (the N sweep).
import 'dart:math';
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

double naiveCosine(Float32List a, Float32List b) {
  var dot = 0.0;
  var na = 0.0;
  var nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (sqrt(na) * sqrt(nb));
}

/// Full scan, score every row, sort, take [k].
List<(int, double)> naiveTopKCosine(VectorMatrix m, Float32List query, int k) {
  final scored = [
    for (var r = 0; r < m.rowCount; r++) (r, naiveCosine(m.rowAt(r), query)),
  ];
  scored.sort((a, b) => b.$2.compareTo(a.$2));
  return scored.sublist(0, min(k, scored.length));
}
