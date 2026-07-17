import 'dart:math';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

import 'helpers.dart';

/// Full-scan reference: scores every row with [score], sorts best first,
/// and returns the first [k]. [descending] is false for distances.
List<(int, double)> refTopK(
  VectorMatrix matrix,
  int k,
  double Function(Float32List row) score, {
  bool descending = true,
}) {
  final scored = [
    for (var r = 0; r < matrix.rowCount; r++) (r, score(matrix.rowAt(r))),
  ];
  scored.sort(
    (a, b) => descending ? b.$2.compareTo(a.$2) : a.$2.compareTo(b.$2),
  );
  return scored.sublist(0, min(k, scored.length));
}

VectorMatrix randomMatrix(int rows, int dim, Random rng) {
  final matrix = VectorMatrix(dim);
  for (var r = 0; r < rows; r++) {
    matrix.add(randomVector(dim, rng));
  }
  return matrix;
}

void expectSameRanking(
  List<(int, double)> got,
  List<(int, double)> expected,
  double tolerance,
) {
  expect(got.length, expected.length);
  for (var i = 0; i < got.length; i++) {
    expect(got[i].$1, expected[i].$1, reason: 'rank $i');
    expect(got[i].$2, closeTo(expected[i].$2, tolerance), reason: 'rank $i');
  }
}

void main() {
  group('construction', () {
    test('rejects a non-positive dimension', () {
      expect(() => VectorMatrix(0), throwsArgumentError);
      expect(() => VectorMatrix(-3), throwsArgumentError);
    });

    test('fromRows stores every row in order', () {
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      expect(matrix.dimension, 3);
      expect(matrix.rowCount, 2);
      expect(matrix.rowAt(0), [1.0, 2.0, 3.0]);
      expect(matrix.rowAt(1), [4.0, 5.0, 6.0]);
    });

    test('fromRows rejects an empty list and ragged rows', () {
      expect(() => VectorMatrix.fromRows([]), throwsArgumentError);
      expect(
        () => VectorMatrix.fromRows([
          [1.0, 2.0],
          [1.0, 2.0, 3.0],
        ]),
        throwsArgumentError,
      );
    });

    test('fromRows rejects doubles that overflow single precision', () {
      // 1e300 is finite as a double but infinite as a float32.
      expect(
        () => VectorMatrix.fromRows([
          [1e300, 0.0],
        ]),
        throwsArgumentError,
      );
    });
  });

  group('add and rowAt', () {
    test('grows past the initial capacity without losing rows', () {
      final matrix = VectorMatrix(5);
      for (var r = 0; r < 100; r++) {
        matrix.add(
          Float32List.fromList([for (var c = 0; c < 5; c++) r + c / 10]),
        );
      }
      expect(matrix.rowCount, 100);
      for (var r = 0; r < 100; r++) {
        final row = matrix.rowAt(r);
        expect(row.length, 5);
        for (var c = 0; c < 5; c++) {
          expect(row[c], closeTo(r + c / 10, 1e-5), reason: 'row $r col $c');
        }
      }
    });

    test('rejects rows of the wrong length', () {
      final matrix = VectorMatrix(4);
      expect(() => matrix.add(Float32List(3)), throwsArgumentError);
      expect(matrix.rowCount, 0);
    });

    test('rejects non-finite rows without changing rowCount', () {
      final matrix = VectorMatrix(3);
      matrix.add(Float32List.fromList([1, 2, 3]));
      expect(
        () => matrix.add(Float32List.fromList([1, double.nan, 3])),
        throwsArgumentError,
      );
      expect(matrix.rowCount, 1);
      // The slot is reusable after the failed add.
      matrix.add(Float32List.fromList([4, 5, 6]));
      expect(matrix.rowAt(1), [4.0, 5.0, 6.0]);
    });

    test('rejects rows whose squared norm overflows single precision', () {
      final matrix = VectorMatrix(2);
      expect(
        () => matrix.add(Float32List.fromList([3e38, 0])),
        throwsArgumentError,
      );
      expect(matrix.rowCount, 0);
    });

    test('rowAt rejects out-of-range indexes', () {
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0],
      ]);
      expect(() => matrix.rowAt(-1), throwsRangeError);
      expect(() => matrix.rowAt(1), throwsRangeError);
    });

    test('rowAt returns a live view into the storage', () {
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0, 3.0],
      ]);
      final view = matrix.rowAt(0);
      view[0] = 9;
      expect(matrix.rowAt(0)[0], 9.0);
      expect(view, matrix.rowAt(0));
    });

    test('padding never leaks into the exposed row', () {
      // Dimension 3 pads each row to 4 floats internally.
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      expect(matrix.rowAt(0).length, 3);
      expect(matrix.rowAt(1), [4.0, 5.0, 6.0]);
    });
  });

  group('top-k search', () {
    test('topKCosine ranks a hand-computed example', () {
      final matrix = VectorMatrix.fromRows([
        [0.0, 1.0], // orthogonal to the query
        [1.0, 0.0], // identical direction
        [1.0, 1.0], // 45 degrees
      ]);
      final result = matrix.topKCosine(Float32List.fromList([2, 0]), 2);
      expect(result.length, 2);
      expect(result[0].$1, 1);
      expect(result[0].$2, closeTo(1, 1e-6));
      expect(result[1].$1, 2);
      expect(result[1].$2, closeTo(1 / sqrt2, 1e-6));
    });

    test('topKCosine matches a full-sort reference on 1000 random '
        'rows', () {
      final rng = Random(10);
      final matrix = randomMatrix(1000, 32, rng);
      final query = randomVector(32, rng);
      final got = matrix.topKCosine(query, 10);
      final expected = refTopK(matrix, 10, (row) => refCosine(row, query));
      expectSameRanking(got, expected, 1e-5);
    });

    test('topKCosine handles dimensions with padding', () {
      final rng = Random(11);
      for (final dim in [5, 7, 33]) {
        final matrix = randomMatrix(200, dim, rng);
        final query = randomVector(dim, rng);
        final got = matrix.topKCosine(query, 5);
        final expected = refTopK(matrix, 5, (row) => refCosine(row, query));
        expectSameRanking(got, expected, 1e-5);
      }
    });

    test('topKDot matches a full-sort reference', () {
      final rng = Random(12);
      final matrix = randomMatrix(1000, 32, rng);
      final query = randomVector(32, rng);
      final got = matrix.topKDot(query, 10);
      final expected = refTopK(matrix, 10, (row) => refDot(row, query));
      expectSameRanking(got, expected, 1e-5);
    });

    test('topKEuclidean returns the nearest rows, nearest first', () {
      final rng = Random(13);
      final matrix = randomMatrix(1000, 32, rng);
      final query = randomVector(32, rng);
      final got = matrix.topKEuclidean(query, 10);
      final expected = refTopK(
        matrix,
        10,
        (row) => refDistance(row, query),
        descending: false,
      );
      expectSameRanking(got, expected, 1e-5);
    });

    test('scores come back ordered', () {
      final rng = Random(14);
      final matrix = randomMatrix(300, 17, rng);
      final query = randomVector(17, rng);
      for (final result in [
        matrix.topKCosine(query, 20),
        matrix.topKDot(query, 20),
      ]) {
        for (var i = 1; i < result.length; i++) {
          expect(result[i].$2, lessThanOrEqualTo(result[i - 1].$2));
        }
      }
      final nearest = matrix.topKEuclidean(query, 20);
      for (var i = 1; i < nearest.length; i++) {
        expect(nearest[i].$2, greaterThanOrEqualTo(nearest[i - 1].$2));
      }
    });

    test('k larger than rowCount returns rowCount entries', () {
      final rng = Random(15);
      final matrix = randomMatrix(4, 8, rng);
      final query = randomVector(8, rng);
      expect(matrix.topKCosine(query, 100).length, 4);
      expect(matrix.topKDot(query, 100).length, 4);
      expect(matrix.topKEuclidean(query, 100).length, 4);
    });

    test('rejects non-positive k and bad queries', () {
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0],
      ]);
      final query = Float32List.fromList([1, 1]);
      expect(() => matrix.topKCosine(query, 0), throwsArgumentError);
      expect(() => matrix.topKDot(query, -1), throwsArgumentError);
      expect(() => matrix.topKCosine(Float32List(3), 1), throwsArgumentError);
      expect(
        () => matrix.topKDot(Float32List.fromList([1, double.nan]), 1),
        throwsArgumentError,
      );
    });

    test('topKCosine rejects a zero query and skips zero rows', () {
      final matrix = VectorMatrix.fromRows([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
      ]);
      final query = Float32List.fromList([1, 1]);
      expect(() => matrix.topKCosine(Float32List(2), 1), throwsArgumentError);
      final result = matrix.topKCosine(query, 3);
      expect(result.length, 2);
      expect(result.map((e) => e.$1), isNot(contains(0)));
    });

    test('topKDot allows a zero query', () {
      final matrix = VectorMatrix.fromRows([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final result = matrix.topKDot(Float32List(2), 2);
      expect(result.length, 2);
      expect(result[0].$2, 0.0);
      expect(result[1].$2, 0.0);
    });

    test('searches on an empty matrix return an empty list', () {
      final matrix = VectorMatrix(4);
      final query = Float32List.fromList([1, 0, 0, 0]);
      expect(matrix.topKCosine(query, 3), isEmpty);
      expect(matrix.topKDot(query, 3), isEmpty);
      expect(matrix.topKEuclidean(query, 3), isEmpty);
    });
  });

  group('toBytes and fromBytes', () {
    test('round-trips exactly, including padded dimensions', () {
      final rng = Random(16);
      for (final dim in [3, 4, 7, 16]) {
        final matrix = randomMatrix(10, dim, rng);
        final restored = VectorMatrix.fromBytes(matrix.toBytes());
        expect(restored.dimension, dim);
        expect(restored.rowCount, 10);
        for (var r = 0; r < 10; r++) {
          expect(restored.rowAt(r), matrix.rowAt(r), reason: 'dim $dim row $r');
        }
      }
    });

    test('a restored matrix searches identically', () {
      final rng = Random(17);
      final matrix = randomMatrix(100, 9, rng);
      final restored = VectorMatrix.fromBytes(matrix.toBytes());
      final query = randomVector(9, rng);
      expect(restored.topKCosine(query, 5), matrix.topKCosine(query, 5));
      expect(restored.topKEuclidean(query, 5), matrix.topKEuclidean(query, 5));
    });

    test('round-trips an empty matrix', () {
      final matrix = VectorMatrix(6);
      final bytes = matrix.toBytes();
      expect(bytes.length, 12);
      final restored = VectorMatrix.fromBytes(bytes);
      expect(restored.dimension, 6);
      expect(restored.rowCount, 0);
    });

    test('writes the documented little-endian layout', () {
      final matrix = VectorMatrix.fromRows([
        [1.0],
      ]);
      final bytes = matrix.toBytes();
      expect(String.fromCharCodes(bytes.sublist(0, 4)), 'VKT1');
      expect(bytes.sublist(4, 8), [1, 0, 0, 0]); // dimension = 1
      expect(bytes.sublist(8, 12), [1, 0, 0, 0]); // rowCount = 1
      expect(bytes.sublist(12), [0, 0, 0x80, 0x3F]); // 1.0f, little-endian
    });

    test('rejects a bad magic', () {
      final bytes = VectorMatrix.fromRows([
        [1.0],
      ]).toBytes();
      bytes[0] = 0x58;
      expect(() => VectorMatrix.fromBytes(bytes), throwsFormatException);
    });

    test('rejects a truncated header', () {
      expect(
        () => VectorMatrix.fromBytes(Uint8List.fromList([0x56, 0x4B, 0x54])),
        throwsFormatException,
      );
    });

    test('rejects truncated and oversized payloads', () {
      final bytes = VectorMatrix.fromRows([
        [1.0, 2.0, 3.0],
      ]).toBytes();
      expect(
        () => VectorMatrix.fromBytes(Uint8List.sublistView(bytes, 0, 20)),
        throwsFormatException,
      );
      final padded = Uint8List(bytes.length + 4)
        ..setRange(0, bytes.length, bytes);
      expect(() => VectorMatrix.fromBytes(padded), throwsFormatException);
    });

    test('rejects a zero dimension', () {
      final bytes = Uint8List(12);
      bytes.setRange(0, 4, 'VKT1'.codeUnits);
      // dimension = 0, rowCount = 0.
      expect(() => VectorMatrix.fromBytes(bytes), throwsFormatException);
    });

    test('rejects non-finite payload components', () {
      final bytes = VectorMatrix.fromRows([
        [1.0, 2.0],
      ]).toBytes();
      // Overwrite the second float with NaN (0x7FC00000, little-endian).
      bytes.setRange(16, 20, [0x00, 0x00, 0xC0, 0x7F]);
      expect(() => VectorMatrix.fromBytes(bytes), throwsFormatException);
    });
  });
}
