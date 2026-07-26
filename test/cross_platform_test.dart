import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

/// Pins the results to the same values on every target.
///
/// The kernels differ by platform: `Float32x4` on the Dart VM, plain scalar
/// code everywhere else, because the emulated `Float32x4` off the VM is slower
/// than no SIMD at all. The VM therefore accumulates in float32 and the web
/// accumulates in double, so the two produce slightly different sums.
///
/// The size of that difference is the thing that matters, and it is not
/// something to assume. These expectations were produced on the VM and are
/// asserted unchanged on the web, so a divergence in either the returned rows
/// or their order fails the suite rather than reaching a user. Scores are
/// compared with a tolerance because they legitimately differ; rows and
/// ordering are compared exactly because they must not.
void main() {
  const rows = 500;
  const dimension = 128;
  const k = 10;

  /// Deterministic on every platform: integer arithmetic only, no `Random`,
  /// no transcendental functions.
  ///
  /// Row magnitudes deliberately span 0.25x to 4x. An earlier version of this
  /// corpus gave every row roughly the same norm, and cosine, dot and
  /// euclidean then returned identical rankings, which would have hidden a
  /// fault in any one of them.
  VectorMatrix buildMatrix() {
    final m = VectorMatrix(dimension);
    for (var r = 0; r < rows; r++) {
      final scale = 0.25 + (r % 16) * 0.25;
      final row = Float32List(dimension);
      for (var i = 0; i < dimension; i++) {
        row[i] = ((((r * 37 + i * 11) * 6151) % 2003) / 2003.0 - 0.5) * scale;
      }
      m.add(row);
    }
    return m;
  }

  Float32List buildQuery(int seed) {
    final q = Float32List(dimension);
    for (var i = 0; i < dimension; i++) {
      q[i] = (((seed * 29 + i * 7) * 4093) % 1741) / 1741.0 - 0.5;
    }
    return q;
  }

  test('top-k rows and their order are identical on every platform', () {
    final m = buildMatrix();
    // Produced by running this file on the Dart VM. If the web kernels ever
    // rank differently, these fail.
    const expected = <String, List<int>>{
      'cosine': [440, 310, 180, 50, 94, 415, 285, 155, 371, 241],
      'dot': [415, 111, 94, 47, 285, 238, 188, 382, 108, 155],
      'euclidean': [432, 368, 224, 448, 288, 0, 352, 304, 384, 208],
    };
    final actual = {
      'cosine': m.topKCosine(buildQuery(1), k).map((e) => e.$1).toList(),
      'dot': m.topKDot(buildQuery(1), k).map((e) => e.$1).toList(),
      'euclidean': m.topKEuclidean(buildQuery(1), k).map((e) => e.$1).toList(),
    };
    // ignore: avoid_print
    print('cross-platform top-$k: $actual');
    expect(actual, expected);
  });

  test('scores agree across platforms to within single-precision error', () {
    final m = buildMatrix();
    // Also produced on the VM. The web accumulates in double and so lands a
    // little closer to the exact sum; 1e-6 relative is the band that difference
    // has to stay inside for the ranking guarantee above to be meaningful.
    const expectedCosine = <double>[
      0.1738678079298492,
      0.1696047031804237,
      0.16195353798926573,
    ];
    final scores = <double>[];
    for (var q = 1; q <= 3; q++) {
      scores.add(m.topKCosine(buildQuery(q), 1).first.$2);
    }
    // ignore: avoid_print
    print('cross-platform score: $scores');
    for (var i = 0; i < scores.length; i++) {
      expect(
        (scores[i] - expectedCosine[i]).abs(),
        lessThan(1e-6),
        reason: 'score $i drifted beyond single-precision error',
      );
    }
  });

  test(
    'a squared norm too large for float32 is rejected on every platform',
    () {
      // The VM accumulates in float32 and reports overflow as a non-finite sum.
      // The web accumulates in double, which would carry this finitely, so the
      // scalar kernels fold out-of-range totals back to infinity. Without that
      // fold this row would be accepted on the web and rejected on the VM.
      final m = VectorMatrix(4);
      final huge = Float32List.fromList([3.0e38, 3.0e38, 3.0e38, 3.0e38]);
      expect(() => m.add(huge), throwsArgumentError);
      expect(m.rowCount, 0);
    },
  );

  test('a non-finite component is still located exactly', () {
    final m = VectorMatrix(8);
    final bad = Float32List(8);
    bad[5] = double.nan;
    expect(
      () => m.add(bad),
      throwsA(
        isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('component 5'),
        ),
      ),
    );
  });
}
