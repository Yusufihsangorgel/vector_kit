// Benchmark for the numbers quoted in the README.
//
// Run with:
//   dart run bench/bench.dart
//
// Baselines are deliberately the code a user would write without this
// package: a scalar loop over List<double>, over Float64List, and over
// Float32List, plus a full-scan-and-sort top-k. Inputs rotate through a
// small pool so the JIT cannot fold repeated calls, and every result
// feeds a checksum that is printed at the end.

import 'dart:math';
import 'dart:typed_data';

import 'package:vector_kit/vector_kit.dart';

const dim = 768;
const dotCalls = 1000000;
const pool = 8;
const topKQueries = 20;
const k = 10;

double naiveDotList(List<double> a, List<double> b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

double naiveDotFloat64(Float64List a, Float64List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

double naiveDotFloat32(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

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

/// Full scan, score every row, sort, take k: the baseline top-k.
List<(int, double)> naiveTopKCosine(VectorMatrix m, Float32List query) {
  final scored = [
    for (var r = 0; r < m.rowCount; r++) (r, naiveCosine(m.rowAt(r), query)),
  ];
  scored.sort((a, b) => b.$2.compareTo(a.$2));
  return scored.sublist(0, min(k, scored.length));
}

double runDotVariant(
  String label,
  double Function(int i) call,
  double naiveMs,
) {
  // Warm up so the JIT optimizes the hot path before timing.
  var sink = 0.0;
  for (var i = 0; i < dotCalls ~/ 10; i++) {
    sink += call(i);
  }
  final watch = Stopwatch()..start();
  for (var i = 0; i < dotCalls; i++) {
    sink += call(i);
  }
  watch.stop();
  final ms = watch.elapsedMicroseconds / 1000;
  final nsPerCall = watch.elapsedMicroseconds * 1000 / dotCalls;
  final speedup = naiveMs > 0 ? naiveMs / ms : 1.0;
  print(
    '  ${label.padRight(34)} ${ms.toStringAsFixed(0).padLeft(6)} ms '
    '${nsPerCall.toStringAsFixed(0).padLeft(6)} ns/call '
    '${speedup.toStringAsFixed(2).padLeft(6)}x  (checksum $sink)',
  );
  return ms;
}

void benchDot() {
  print('dot product, $dim-dim, $dotCalls calls, $pool input pairs:');
  final rng = Random(42);
  final aList = [
    for (var p = 0; p < pool; p++)
      [for (var i = 0; i < dim; i++) rng.nextDouble() * 2 - 1],
  ];
  final bList = [
    for (var p = 0; p < pool; p++)
      [for (var i = 0; i < dim; i++) rng.nextDouble() * 2 - 1],
  ];
  final a64 = [for (final v in aList) Float64List.fromList(v)];
  final b64 = [for (final v in bList) Float64List.fromList(v)];
  final a32 = [for (final v in aList) Float32List.fromList(v)];
  final b32 = [for (final v in bList) Float32List.fromList(v)];

  final naiveMs = runDotVariant(
    'scalar List<double>',
    (i) => naiveDotList(aList[i & (pool - 1)], bList[i & (pool - 1)]),
    0,
  );
  runDotVariant(
    'scalar Float64List',
    (i) => naiveDotFloat64(a64[i & (pool - 1)], b64[i & (pool - 1)]),
    naiveMs,
  );
  runDotVariant(
    'scalar Float32List',
    (i) => naiveDotFloat32(a32[i & (pool - 1)], b32[i & (pool - 1)]),
    naiveMs,
  );
  runDotVariant(
    'vector_kit dot (SIMD)',
    (i) => dot(a32[i & (pool - 1)], b32[i & (pool - 1)]),
    naiveMs,
  );
  print('');
}

void benchTopK(int rows) {
  final rng = Random(7);
  final matrix = VectorMatrix(dim);
  for (var r = 0; r < rows; r++) {
    matrix.add(
      Float32List.fromList([
        for (var i = 0; i < dim; i++) rng.nextDouble() * 2 - 1,
      ]),
    );
  }
  final queries = [
    for (var q = 0; q < topKQueries; q++)
      Float32List.fromList([
        for (var i = 0; i < dim; i++) rng.nextDouble() * 2 - 1,
      ]),
  ];

  // Warm up both paths and check that they agree on the winners.
  var agreement = 0;
  for (final query in queries.take(3)) {
    final naive = naiveTopKCosine(matrix, query);
    final simd = matrix.topKCosine(query, k);
    for (var i = 0; i < k; i++) {
      if (naive[i].$1 == simd[i].$1) agreement++;
    }
  }

  var checksum = 0;
  final naiveWatch = Stopwatch()..start();
  for (final query in queries) {
    checksum += naiveTopKCosine(matrix, query).first.$1;
  }
  naiveWatch.stop();

  final simdWatch = Stopwatch()..start();
  for (final query in queries) {
    checksum += matrix.topKCosine(query, k).first.$1;
  }
  simdWatch.stop();

  final naiveMs = naiveWatch.elapsedMicroseconds / 1000 / topKQueries;
  final simdMs = simdWatch.elapsedMicroseconds / 1000 / topKQueries;
  print(
    'topKCosine k=$k, $rows x $dim, $topKQueries queries '
    '(rank agreement $agreement/${3 * k}):',
  );
  print(
    '  ${'full scan + sort'.padRight(34)} '
    '${naiveMs.toStringAsFixed(2).padLeft(8)} ms/query',
  );
  print(
    '  ${'vector_kit topKCosine'.padRight(34)} '
    '${simdMs.toStringAsFixed(2).padLeft(8)} ms/query '
    '${(naiveMs / simdMs).toStringAsFixed(2).padLeft(6)}x  '
    '(checksum $checksum)',
  );
  print('');
}

void main() {
  benchDot();
  benchTopK(10000);
  benchTopK(100000);
}
