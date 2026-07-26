@Tags(['bench'])
library;

import 'dart:math';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

/// Measures what this package costs on the platform it is run on.
///
/// Run it both ways and compare:
///
///     dart test test/platform_cost_test.dart -t bench
///     dart test test/platform_cost_test.dart -t bench -p chrome
///
/// It exists because `Float32x4` is a real SIMD type only on the native VM.
/// Everywhere else it is emulated, and the emulation is slower than not using
/// it at all: the same kernel that wins 2.2x natively loses 11x on the web.
/// The package advertises `platform:web`, so the size of that gap is a fact
/// users are entitled to, and one that should not be able to drift unnoticed.
///
/// The third benchmark is the honest baseline for the other two. A kernel
/// microbenchmark over one cache-resident pair of vectors is not a search: it
/// has no streaming memory traffic, no per-row divide and no top-k to
/// maintain. So this one runs the whole thing a caller would write by hand,
/// over a real packed corpus, and that is the number to compare against
/// `topKCosine` before taking the dependency.
///
/// This asserts nothing about absolute time: CI runners vary too much for a
/// threshold to mean anything. It prints, and the numbers in
/// `doc/web-performance.md` come from running it.
void main() {
  const rows = 1000;
  const dimension = 384;
  const iterations = 200;

  test('topKCosine, 1000 x 384', () {
    final matrix = VectorMatrix(dimension);
    for (var r = 0; r < rows; r++) {
      matrix.add(
        Float32List.fromList(
          List<double>.generate(
            dimension,
            (i) => (((r * 31 + i) * 7919) % 1000) / 1000.0 - 0.5,
          ),
        ),
      );
    }
    final query = Float32List.fromList(
      List<double>.generate(dimension, (i) => (i % 17) / 17.0 - 0.5),
    );

    for (var i = 0; i < 20; i++) {
      matrix.topKCosine(query, 10);
    }

    final watch = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      matrix.topKCosine(query, 10);
    }
    watch.stop();

    // ignore: avoid_print
    print(
      'topKCosine ${rows}x$dimension: '
      '${(watch.elapsedMicroseconds / iterations).toStringAsFixed(1)} us/query',
    );
  });

  test('the kernel choice itself: SIMD versus a plain loop', () {
    // Isolates the question the conditional-kernel work turns on. Same dot
    // product, same data, two implementations.
    final a = Float32List(dimension);
    final b = Float32List(dimension);
    for (var i = 0; i < dimension; i++) {
      a[i] = (i % 13) / 13.0;
      b[i] = (i % 7) / 7.0;
    }
    final aLanes = Float32x4List.view(a.buffer);
    final bLanes = Float32x4List.view(b.buffer);
    final lanes = dimension >> 2;

    double simd() {
      var acc = Float32x4.zero();
      for (var i = 0; i < lanes; i++) {
        acc += aLanes[i] * bLanes[i];
      }
      return acc.x + acc.y + acc.z + acc.w;
    }

    double scalar() {
      var sum = 0.0;
      for (var i = 0; i < dimension; i++) {
        sum += a[i] * b[i];
      }
      return sum;
    }

    for (var i = 0; i < 200; i++) {
      simd();
      scalar();
    }

    var watch = Stopwatch()..start();
    for (var r = 0; r < rows * 20; r++) {
      simd();
    }
    watch.stop();
    final simdUs = watch.elapsedMicroseconds / 20;

    watch = Stopwatch()..start();
    for (var r = 0; r < rows * 20; r++) {
      scalar();
    }
    watch.stop();
    final scalarUs = watch.elapsedMicroseconds / 20;

    // ignore: avoid_print
    print(
      'kernel over $rows rows: SIMD ${simdUs.toStringAsFixed(0)} us, '
      'scalar ${scalarUs.toStringAsFixed(0)} us, '
      'SIMD/scalar ${(simdUs / scalarUs).toStringAsFixed(2)}x',
    );
  });

  test('the baseline: a hand-written scalar scan, no package at all', () {
    // What a caller writes instead of taking this dependency. One packed
    // Float32List, cached row norms, a k-sized insertion top-k. Compare its
    // number against the first benchmark to see what the package actually
    // buys on the platform you are on.
    final data = Float32List(rows * dimension);
    final norms = Float32List(rows);
    for (var i = 0; i < rows; i++) {
      var sumSquares = 0.0;
      for (var j = 0; j < dimension; j++) {
        final v = (((i * 31 + j) * 7919) % 1000) / 1000.0 - 0.5;
        data[i * dimension + j] = v;
        sumSquares += v * v;
      }
      norms[i] = sqrt(sumSquares);
    }
    final query = Float32List(dimension);
    var querySquares = 0.0;
    for (var j = 0; j < dimension; j++) {
      query[j] = (j % 17) / 17.0 - 0.5;
      querySquares += query[j] * query[j];
    }
    final queryNorm = sqrt(querySquares);

    const k = 10;
    final bestIndex = List<int>.filled(k, -1);
    final bestScore = List<double>.filled(k, -2);

    void scan() {
      bestIndex.fillRange(0, k, -1);
      bestScore.fillRange(0, k, -2);
      var base = 0;
      for (var i = 0; i < rows; i++) {
        var dot = 0.0;
        for (var j = 0; j < dimension; j++) {
          dot += data[base + j] * query[j];
        }
        base += dimension;
        final score = dot / (norms[i] * queryNorm);
        if (score > bestScore[k - 1]) {
          var at = k - 1;
          while (at > 0 && bestScore[at - 1] < score) {
            bestScore[at] = bestScore[at - 1];
            bestIndex[at] = bestIndex[at - 1];
            at--;
          }
          bestScore[at] = score;
          bestIndex[at] = i;
        }
      }
    }

    for (var i = 0; i < 20; i++) {
      scan();
    }

    var fastest = double.infinity;
    for (var batch = 0; batch < 10; batch++) {
      final watch = Stopwatch()..start();
      for (var i = 0; i < iterations; i++) {
        scan();
      }
      watch.stop();
      final perQuery = watch.elapsedMicroseconds / iterations;
      if (perQuery < fastest) fastest = perQuery;
    }

    // The top hit is printed so a run on another platform can be checked to
    // have done the same work, not just spent the same time.
    // ignore: avoid_print
    print(
      'hand-written scan ${rows}x$dimension: '
      '${fastest.toStringAsFixed(1)} us/query, top1=${bestIndex[0]}',
    );
  });
}
