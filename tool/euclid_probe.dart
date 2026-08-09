// TEMPORARY probe. Answers one question: can Euclidean distance be recovered
// from the quantized form, and how accurately?
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

/// Replicates QuantizedMatrix.from exactly (vector_matrix.dart:497-519).
({List<double> dequantized, double scale, double norm}) quantizeRow(
  Float32List row,
) {
  final dimension = row.length;
  var maxAbs = 0.0;
  for (var i = 0; i < dimension; i++) {
    final a = row[i].abs();
    if (a > maxAbs) maxAbs = a;
  }
  final scale = maxAbs == 0 ? 0.0 : maxAbs / 127.0;
  final dequantized = <double>[];
  var sum2 = 0.0;
  for (var i = 0; i < dimension; i++) {
    final q = scale == 0 ? 0 : (row[i] / scale).round().clamp(-127, 127);
    final d = q * scale;
    dequantized.add(d);
    sum2 += d * d;
  }
  return (dequantized: dequantized, scale: scale, norm: math.sqrt(sum2));
}

void main() {
  final rng = math.Random(7);
  const dimension = 384;
  const rows = 200;

  final source = <List<double>>[
    for (var r = 0; r < rows; r++)
      [for (var c = 0; c < dimension; c++) rng.nextDouble() * 2 - 1],
  ];
  final matrix = VectorMatrix.fromRows(source);
  final quantized = QuantizedMatrix.from(matrix);
  final query = <double>[
    for (var c = 0; c < dimension; c++) rng.nextDouble() * 2 - 1,
  ];

  var qn2 = 0.0;
  for (final v in query) {
    qn2 += v * v;
  }
  final qNorm = math.sqrt(qn2);

  // ---- 1. Is the information present? Recover distance two ways. ----
  var worstExpansionError = 0.0;
  var worstPublicApiError = 0.0;
  var minDistance = double.infinity;

  // Everything QuantizedMatrix exposes today, for every row.
  final dotScores = {for (final (i, s) in quantized.topKDot(query, rows)) i: s};
  final cosScores = {
    for (final (i, s) in quantized.topKCosine(query, rows)) i: s,
  };

  for (var r = 0; r < rows; r++) {
    final q = quantizeRow(matrix.rowAt(r));

    // Reference: the direct loop, in double. This is what a new
    // topKEuclidean would compute.
    var direct = 0.0;
    for (var i = 0; i < dimension; i++) {
      final d = query[i] - q.dequantized[i];
      direct += d * d;
    }
    final reference = math.sqrt(direct);
    minDistance = math.min(minDistance, reference);

    // (a) Expansion using the values the class already caches internally:
    //     ||y||^2 - 2<y,x̂> + ||x̂||^2
    var dotYX = 0.0;
    for (var i = 0; i < dimension; i++) {
      dotYX += query[i] * q.dequantized[i];
    }
    final expansion = math.sqrt(
      math.max(0.0, qn2 - 2 * dotYX + q.norm * q.norm),
    );

    // (b) Expansion using ONLY today's public API. ||x̂|| is recoverable
    //     because cosine = <y,x̂> / (||y||*||x̂||).
    final dotPub = dotScores[r]!;
    final cosPub = cosScores[r]!;
    final normPub = dotPub / (cosPub * qNorm);
    final publicApi = math.sqrt(
      math.max(0.0, qn2 - 2 * dotPub + normPub * normPub),
    );

    worstExpansionError = math.max(
      worstExpansionError,
      (expansion - reference).abs(),
    );
    worstPublicApiError = math.max(
      worstPublicApiError,
      (publicApi - reference).abs(),
    );
  }

  print('rows=$rows dimension=$dimension');
  print('min distance over the corpus : ${minDistance.toStringAsFixed(6)}');
  print('worst |expansion - direct|   : $worstExpansionError');
  print('worst |publicAPI - direct|   : $worstPublicApiError');

  // ---- 2. The cancellation case the expansion is supposed to fear:
  //         query == the dequantized row, so the true distance is 0. ----
  print('');
  print('--- degenerate case: query IS the row, true distance 0 ---');
  final selfRow = quantizeRow(matrix.rowAt(0));
  var selfDirect = 0.0;
  var selfDot = 0.0;
  var selfN2 = 0.0;
  for (var i = 0; i < dimension; i++) {
    final y = selfRow.dequantized[i];
    selfN2 += y * y;
    selfDot += y * selfRow.dequantized[i];
    final d = y - selfRow.dequantized[i];
    selfDirect += d * d;
  }
  final raw = selfN2 - 2 * selfDot + selfRow.norm * selfRow.norm;
  print('direct squared distance : $selfDirect');
  print('expansion, unclamped    : $raw');
  print('sqrt of that unclamped  : ${math.sqrt(raw)}');

  // ---- 3. Does a zero row survive? Cosine skips it; Euclidean must not. ----
  print('');
  print('--- zero row handling ---');
  final withZero = VectorMatrix.fromRows([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
  ]);
  final qz = QuantizedMatrix.from(withZero);
  final probe = <double>[0.01, 0.01, 0.0, 0.0];
  print('float  topKEuclidean : ${withZero.topKEuclidean(probe, 3)}');
  print('int8   topKCosine    : ${qz.topKCosine(probe, 3)}  <- row 1 skipped');
  print('int8   topKDot       : ${qz.topKDot(probe, 3)}');
}
