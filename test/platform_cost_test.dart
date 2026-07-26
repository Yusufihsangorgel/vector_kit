@Tags(['bench'])
library;

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
/// it at all — the same kernel that wins 2.7x natively loses 9.6x on the web.
/// The package advertises `platform:web`, so the size of that gap is a fact
/// users are entitled to, and one that should not be able to drift unnoticed.
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
}
