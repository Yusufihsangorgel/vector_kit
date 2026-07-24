// What int8 quantization buys and what it costs, measured on the same vectors.
//
//   dart run benchmark/quantization_benchmark.dart
//
// Recall@10 is the fraction of a query's true top-10 (by the float matrix) that
// the quantized matrix also returns in its top-10 — how much of the answer
// survives the rounding. Memory is the stored bytes of each. Time is the search
// itself; quantization trades throughput for memory, and this shows by how much.
import 'dart:math';

import 'package:vector_kit/vector_kit.dart';

const _dimension = 768; // a common sentence-embedding size
const _count = 5000;
const _queries = 200;
const _k = 10;

List<double> _unitVector(Random rng) {
  final v = [for (var i = 0; i < _dimension; i++) rng.nextDouble() * 2 - 1];
  var norm = 0.0;
  for (final x in v) {
    norm += x * x;
  }
  norm = sqrt(norm);
  return [for (final x in v) x / norm];
}

Set<int> _indices(List<(int, double)> hits) => {for (final h in hits) h.$1};

void main() {
  // Fixed seed so the numbers are reproducible run to run.
  final rng = Random(20260724);

  final rows = [for (var i = 0; i < _count; i++) _unitVector(rng)];
  final queries = [for (var i = 0; i < _queries; i++) _unitVector(rng)];

  final matrix = VectorMatrix.fromRows(rows);
  final quantized = QuantizedMatrix.from(matrix);

  // Recall@k: quantized top-k vs the float top-k, averaged over the queries.
  var recallSum = 0.0;
  for (final q in queries) {
    final truth = _indices(matrix.topKCosine(q, _k));
    final got = _indices(quantized.topKCosine(q, _k));
    recallSum += truth.intersection(got).length / _k;
  }
  final recall = recallSum / _queries;

  // Search time for the same set of queries, each side warmed once.
  matrix.topKCosine(queries.first, _k);
  final swFloat = Stopwatch()..start();
  for (final q in queries) {
    matrix.topKCosine(q, _k);
  }
  swFloat.stop();

  quantized.topKCosine(queries.first, _k);
  final swQuant = Stopwatch()..start();
  for (final q in queries) {
    quantized.topKCosine(q, _k);
  }
  swQuant.stop();

  final floatBytes = _count * _dimension * 4;
  final quantBytes = quantized.byteSize;

  String mb(int b) => '${(b / (1024 * 1024)).toStringAsFixed(1)} MB';
  String us(Stopwatch s) =>
      '${(s.elapsedMicroseconds / _queries).toStringAsFixed(0)} µs/query';

  print('$_count rows x $_dimension dims, top-$_k, $_queries queries\n');
  print('memory   float32  ${mb(floatBytes)}');
  print(
    '         int8     ${mb(quantBytes)}   '
    '(${(floatBytes / quantBytes).toStringAsFixed(1)}x smaller)',
  );
  print(
    'recall@$_k         ${(recall * 100).toStringAsFixed(1)}%   '
    'of the float top-$_k survives quantization',
  );
  print('search   float32  ${us(swFloat)}');
  print(
    '         int8     ${us(swQuant)}   '
    '(${(swQuant.elapsedMicroseconds / swFloat.elapsedMicroseconds).toStringAsFixed(1)}x '
    'the float time — memory is the trade, not speed)',
  );
}
