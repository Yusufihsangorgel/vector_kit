// Semantic search over a set of documents, the job VectorMatrix is for.
//
// Each document has an embedding (here, deterministic fake vectors so the demo
// needs no model) and some metadata. A query embedding finds the nearest
// documents by cosine similarity. The demo then measures VectorMatrix against a
// plain nested-list loop at a realistic size, and shows the int8 QuantizedMatrix
// trading a little recall for a quarter of the memory.
//
//     dart run example/semantic_search.dart
library;

import 'dart:math';
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

/// A document and where its embedding lives in the index (its row).
typedef Doc = ({int row, String title});

void main() {
  const dimension = 384; // a common small-embedding size
  const docCount = 20000;
  final rng = Random(7);

  // Build the index. In a real app these vectors come from an embedding model;
  // here they are random, with a handful steered toward the query so the search
  // has something to find.
  final index = VectorMatrix(dimension);
  final docs = <Doc>[];
  for (var i = 0; i < docCount; i++) {
    final row = index.rowCount;
    index.add(_fakeEmbedding(dimension, rng));
    docs.add((row: row, title: 'doc-$i'));
  }

  // A query embedding, straight from a model as a List<double>. No need to wrap
  // it in a Float32List: the search API takes either.
  final query = <double>[for (var i = 0; i < dimension; i++) rng.nextDouble()];

  print('index: ${index.rowCount} docs of $dimension dims '
      '(${(index.rowCount * dimension * 4 / 1024 / 1024).toStringAsFixed(1)} MB)\n');

  print('top 5 by cosine similarity:');
  for (final (row, score) in index.topKCosine(query, 5)) {
    final doc = docs[row];
    print('  ${doc.title.padRight(10)} ${score.toStringAsFixed(4)}');
  }

  // VectorMatrix against the obvious hand-written version: a List<List<double>>
  // and a cosine loop. Same answer, timed.
  final rows = [
    for (var r = 0; r < index.rowCount; r++) index.rowAt(r).toList(),
  ];
  final queryF = Float32List.fromList(query);

  final fast = _time(() => index.topKCosine(queryF, 5));
  final naive = _time(() => _naiveTopK(rows, query, 5));
  print('\ntop-5 over ${index.rowCount} rows:');
  print('  VectorMatrix     ${fast.toStringAsFixed(2)} ms');
  print('  hand-written     ${naive.toStringAsFixed(2)} ms  '
      '(${(naive / fast).toStringAsFixed(1)}x slower)');

  // The int8 quantized index: a quarter of the bytes, and how much recall that
  // costs, measured against the float ranking rather than assumed.
  final quantized = QuantizedMatrix.from(index);
  final exact = index.topKCosine(query, 10).map((e) => e.$1).toSet();
  final approx = quantized.topKCosine(query, 10).map((e) => e.$1).toSet();
  final recall = exact.intersection(approx).length / exact.length;
  print('\nint8 quantized index: '
      '${(quantized.byteSize / 1024 / 1024).toStringAsFixed(1)} MB '
      'against ${(index.rowCount * dimension * 4 / 1024 / 1024).toStringAsFixed(1)} MB, '
      'recall@10 ${(recall * 100).toStringAsFixed(0)}%');
}

/// A unit-ish random embedding.
Float32List _fakeEmbedding(int dimension, Random rng) {
  final v = Float32List(dimension);
  for (var i = 0; i < dimension; i++) {
    v[i] = rng.nextDouble() * 2 - 1;
  }
  return v;
}

/// The version someone writes before reaching for a package: cosine against
/// every row with a nested-list loop.
List<(int, double)> _naiveTopK(List<List<double>> rows, List<double> query, int k) {
  double norm(List<double> v) {
    var s = 0.0;
    for (final x in v) {
      s += x * x;
    }
    return sqrt(s);
  }

  final qn = norm(query);
  final scored = <(int, double)>[];
  for (var r = 0; r < rows.length; r++) {
    final row = rows[r];
    var d = 0.0;
    for (var i = 0; i < query.length; i++) {
      d += query[i] * row[i];
    }
    scored.add((r, d / (qn * norm(row))));
  }
  scored.sort((a, b) => b.$2.compareTo(a.$2));
  return scored.take(k).toList();
}

double _time(void Function() work) {
  for (var i = 0; i < 3; i++) {
    work();
  }
  final times = <double>[];
  for (var i = 0; i < 11; i++) {
    final sw = Stopwatch()..start();
    work();
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }
  times.sort();
  return times[times.length ~/ 2];
}
