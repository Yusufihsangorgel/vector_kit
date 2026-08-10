import 'dart:math';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

/// A reproducible corpus of unit-ish vectors, the shape embeddings have.
VectorMatrix _corpus({int rows = 400, int dimension = 128, int seed = 11}) {
  final rng = Random(seed);
  return VectorMatrix.fromRows([
    for (var r = 0; r < rows; r++)
      [for (var i = 0; i < dimension; i++) rng.nextDouble() * 2 - 1],
  ]);
}

Float32List _query(int dimension, {int seed = 99}) {
  final rng = Random(seed);
  return Float32List.fromList([
    for (var i = 0; i < dimension; i++) rng.nextDouble() * 2 - 1,
  ]);
}

/// Rows that survive quantization untouched.
///
/// Every component is a whole number and every row peaks at exactly 127, so
/// the per-row scale is 1.0 and each stored byte dequantizes back to the
/// value written here. That is what lets the tests below pin an exact
/// ordering and an exact distance instead of a tolerance.
List<List<double>> _losslessRows() => [
  [127.0, 0.0, 0.0, 0.0],
  [127.0, 10.0, 0.0, 0.0],
  [127.0, 0.0, 30.0, 0.0],
  [-127.0, 5.0, 0.0, 0.0],
  [127.0, 0.0, 0.0, 60.0],
];

/// Two 768-component rows of whole numbers, each peaking at exactly 127, so
/// the scale is 1.0 and every stored byte dequantizes back to the value
/// generated here.
List<List<double>> _wholeNumberRows() {
  final rng = Random(4242);
  List<double> row() => [
    127.0,
    for (var i = 1; i < 768; i++) (rng.nextInt(255) - 127).toDouble(),
  ];
  return [row(), row()];
}

/// Distances worked out row by row and sorted, with no part of the search
/// under test involved.
List<(int index, double distance)> _nearestByBruteForce(
  List<List<double>> rows,
  List<double> query,
  int k,
) {
  final scored = <(int, double)>[];
  for (var r = 0; r < rows.length; r++) {
    var sum2 = 0.0;
    for (var i = 0; i < query.length; i++) {
      final d = query[i] - rows[r][i];
      sum2 += d * d;
    }
    scored.add((r, sqrt(sum2)));
  }
  scored.sort((a, b) => a.$2.compareTo(b.$2));
  return scored.take(k).toList();
}

void main() {
  test('it stores about a quarter of the bytes', () {
    final matrix = _corpus();
    final quantized = QuantizedMatrix.from(matrix);

    final floatBytes = matrix.rowCount * matrix.dimension * 4;
    expect(quantized.byteSize, lessThan(floatBytes ~/ 3));
    // 128 dims: 128 bytes a row against 512, plus 16 bytes of scale and norm.
    expect(quantized.rowCount, matrix.rowCount);
  });

  test('recall against the float32 ranking is measured, not assumed', () {
    final matrix = _corpus();
    final quantized = QuantizedMatrix.from(matrix);
    const k = 10;

    var hits = 0;
    var queries = 0;
    for (var s = 0; s < 40; s++) {
      final query = _query(matrix.dimension, seed: 1000 + s);
      final exact = matrix.topKCosine(query, k).map((e) => e.$1).toSet();
      final approximate = quantized.topKCosine(query, k).map((e) => e.$1);
      hits += approximate.where(exact.contains).length;
      queries++;
    }
    final recall = hits / (queries * k);
    // Eight-bit rounding costs something; the point is to know how much. This
    // floor is what the implementation actually reaches, so a regression that
    // degrades accuracy fails here instead of shipping quietly.
    expect(recall, greaterThan(0.95), reason: 'recall@$k was $recall');
  });

  test('scores land close to the exact ones', () {
    final matrix = _corpus(rows: 50, dimension: 64);
    final quantized = QuantizedMatrix.from(matrix);
    final query = _query(64);

    final exact = {for (final e in matrix.topKCosine(query, 50)) e.$1: e.$2};
    for (final (index, score) in quantized.topKCosine(query, 50)) {
      expect(score, closeTo(exact[index]!, 0.01));
    }
  });

  test('the top result usually matches exactly', () {
    final matrix = _corpus();
    final quantized = QuantizedMatrix.from(matrix);
    var same = 0;
    for (var s = 0; s < 40; s++) {
      final query = _query(matrix.dimension, seed: 2000 + s);
      if (matrix.topKCosine(query, 1).single.$1 ==
          quantized.topKCosine(query, 1).single.$1) {
        same++;
      }
    }
    expect(same, greaterThan(35), reason: 'top-1 agreed on $same of 40');
  });

  test('dot product ranking survives quantization', () {
    final matrix = _corpus(rows: 100, dimension: 64);
    final quantized = QuantizedMatrix.from(matrix);
    final query = _query(64, seed: 7);
    expect(
      quantized
          .topKDot(query, 5)
          .map((e) => e.$1)
          .toSet()
          .intersection(matrix.topKDot(query, 5).map((e) => e.$1).toSet()),
      hasLength(greaterThanOrEqualTo(4)),
    );
  });

  test('euclidean ranking survives quantization', () {
    final matrix = _corpus(rows: 100, dimension: 64);
    final quantized = QuantizedMatrix.from(matrix);
    final query = _query(64, seed: 7);
    expect(
      quantized
          .topKEuclidean(query, 5)
          .map((e) => e.$1)
          .toSet()
          .intersection(
            matrix.topKEuclidean(query, 5).map((e) => e.$1).toSet(),
          ),
      hasLength(greaterThanOrEqualTo(4)),
    );
  });

  test('nearest rows come back nearest first', () {
    // The rows are lossless, so the brute-force distances are the distances
    // the search should report rather than an approximation of them. This is
    // the test that catches a sign error: drop the negation that turns
    // "smallest distance" into "largest score" and the farthest rows come
    // back instead, in a well-formed list of exactly the right length.
    final rows = _losslessRows();
    final quantized = QuantizedMatrix.from(VectorMatrix.fromRows(rows));
    final query = <double>[127.0, 2.0, 1.0, 0.0];

    final got = quantized.topKEuclidean(query, rows.length);
    final expected = _nearestByBruteForce(rows, query, rows.length);

    expect(got.map((e) => e.$1).toList(), expected.map((e) => e.$1).toList());
    for (var i = 0; i < got.length; i++) {
      expect(got[i].$2, closeTo(expected[i].$2, 1e-12));
    }
  });

  test('scores are distances, not squared distances', () {
    // The 127 peaks the row, so the scale is exactly 1.0 and the stored row
    // is exactly [127, 0, 0]. The query sits 3 away along one axis and 4
    // along another, which is 5 by Pythagoras and 25 without the square root.
    final quantized = QuantizedMatrix.from(
      VectorMatrix.fromRows([
        [127.0, 0.0, 0.0],
      ]),
    );
    expect(quantized.topKEuclidean([124.0, 4.0, 0.0], 1).single.$2, 5.0);
  });

  test('a zero row is scored, not skipped', () {
    // topKCosine skips a zero row because it has no direction to compare
    // against. A zero row is still a point at the origin, and it can be the
    // nearest one, so the Euclidean search has to score it.
    final matrix = VectorMatrix.fromRows([
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 3.0],
    ]);
    final quantized = QuantizedMatrix.from(matrix);
    final results = quantized.topKEuclidean([1.0, 2.0, 3.0], 2);

    expect(results.map((e) => e.$1).toList(), [1, 0]);
    expect(results[1].$2, closeTo(sqrt(14), 1e-12));
  });

  test('a zero query is allowed and ranks by row norm', () {
    // Cosine throws here because there is no angle to measure. The distance
    // from the origin to a row is just that row's norm, which is a fair
    // question to ask, so the Euclidean search answers it.
    final rows = <List<double>>[
      [127.0, 0.0, 0.0],
      [127.0, 127.0, 0.0],
      [127.0, 127.0, 127.0],
    ];
    final quantized = QuantizedMatrix.from(VectorMatrix.fromRows(rows));
    final origin = <double>[0.0, 0.0, 0.0];

    expect(() => quantized.topKCosine(origin, 3), throwsArgumentError);

    final results = quantized.topKEuclidean(origin, 3);
    expect(results.map((e) => e.$1).toList(), [0, 1, 2]);
    expect(results[0].$2, closeTo(127.0, 1e-12));
    expect(results[2].$2, closeTo(sqrt(3) * 127.0, 1e-12));
  });

  test('k larger than rowCount returns rowCount entries', () {
    // The zero row does not shorten the result the way it does for cosine.
    final matrix = VectorMatrix.fromRows([
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
    final quantized = QuantizedMatrix.from(matrix);
    final query = <double>[1.0, 1.0, 1.0];
    expect(quantized.topKEuclidean(query, 10).length, 3);
    expect(quantized.topKCosine(query, 10).length, 2);
  });

  test('a euclidean search on an empty matrix returns an empty list', () {
    final quantized = QuantizedMatrix.from(VectorMatrix(4));
    expect(quantized.topKEuclidean([1.0, 0.0, 0.0, 0.0], 3), isEmpty);
  });

  test('scores come back ordered, nearest first', () {
    final quantized = QuantizedMatrix.from(_corpus(rows: 200, dimension: 32));
    for (var s = 0; s < 10; s++) {
      final results = quantized.topKEuclidean(_query(32, seed: 300 + s), 10);
      expect(results, hasLength(10));
      for (var i = 1; i < results.length; i++) {
        expect(
          results[i].$2,
          greaterThanOrEqualTo(results[i - 1].$2),
          reason: 'query $s: ${results[i - 1]} came before ${results[i]}',
        );
      }
    }
  });

  test('a query equal to a stored row scores exactly zero', () {
    // The rows round-trip, so the stored row is the row generated here and
    // the distance from it to itself has no rounding to hide behind.
    final rows = _wholeNumberRows();
    final quantized = QuantizedMatrix.from(VectorMatrix.fromRows(rows));

    final results = quantized.topKEuclidean(rows[0], 2);
    expect(results.first.$1, 0);
    expect(results.first.$2, 0.0);
  });

  test('a near miss keeps its precision', () {
    // This is what holds the distance to a direct loop over the differences.
    // Expanding it to |q|^2 - 2<q,x> + |x|^2 would let the dot loop and the
    // cached norm be reused, but it reaches a small distance by subtracting
    // numbers near 4.1e6: for the query below it returns 1.00012e-3 against
    // a true separation of 1e-3, while the direct loop lands within 2.4e-15
    // of it.
    //
    // Measured, not assumed: the exact-zero test above does not catch this by
    // itself. When the query is the row, the components are whole numbers, so
    // |q|^2 is exact and the cached norm usually squares back to it, and the
    // expansion reaches exactly 0.0 too. Over 500 seeds of a corpus shaped
    // like this one it did so in 265 of them, which is enough that a test
    // checking only the zero case would have passed with either
    // implementation. A near miss separates them every time, and a near miss
    // is what this search is made of.
    final rows = _wholeNumberRows();
    final quantized = QuantizedMatrix.from(VectorMatrix.fromRows(rows));

    final query = List<double>.from(rows[0]);
    query[5] += 0.001;

    final results = quantized.topKEuclidean(query, 1);
    expect(results.single.$1, 0);
    expect(results.single.$2, closeTo(0.001, 1e-9));
  });

  test('a zero row is skipped rather than scored', () {
    final matrix = VectorMatrix.fromRows([
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 3.0],
    ]);
    final quantized = QuantizedMatrix.from(matrix);
    final results = quantized.topKCosine(Float32List.fromList([1, 2, 3]), 2);
    expect(results.map((e) => e.$1), [1]);
  });

  test('the source matrix is left alone so the two can be compared', () {
    final matrix = _corpus(rows: 5, dimension: 8);
    final before = matrix.rowAt(0).toList();
    QuantizedMatrix.from(matrix);
    expect(matrix.rowAt(0).toList(), before);
  });

  test('cosine stays inside [-1, 1]', () {
    // The norm cached per row is the norm of what is stored, not of the row it
    // came from. Taking it from the original instead pushes scores past 1, and
    // topKCosine here has no clamp to hide it: measured at 1.0010 with a
    // 20-row corpus when the norm is computed from the source row.
    final matrix = _corpus();
    final quantized = QuantizedMatrix.from(matrix);

    for (var r = 0; r < matrix.rowCount; r++) {
      final query = matrix.rowAt(r).toList();
      for (final (_, score) in quantized.topKCosine(query, 3)) {
        expect(score, lessThanOrEqualTo(1.0));
        expect(score, greaterThanOrEqualTo(-1.0));
      }
    }
  });

  test('bad arguments are rejected', () {
    final quantized = QuantizedMatrix.from(_corpus(rows: 4, dimension: 8));
    expect(() => quantized.topKCosine(Float32List(8), 0), throwsArgumentError);
    expect(() => quantized.topKCosine(Float32List(7), 1), throwsArgumentError);
    // An all-zero query has no direction to compare against.
    expect(() => quantized.topKCosine(Float32List(8), 1), throwsArgumentError);
    expect(
      () => quantized.topKEuclidean(Float32List(8), 0),
      throwsArgumentError,
    );
    expect(
      () => quantized.topKEuclidean(Float32List(7), 1),
      throwsArgumentError,
    );
  });

  test('rejects a NaN or infinite component in the query', () {
    final quantized = QuantizedMatrix.from(_corpus(rows: 4, dimension: 8));
    expect(
      () => quantized.topKCosine(
        Float32List.fromList([1, 2, 3, 4, double.nan, 6, 7, 8]),
        1,
      ),
      throwsArgumentError,
    );
    expect(
      () => quantized.topKDot(
        Float32List.fromList([double.infinity, 2, 3, 4, 5, 6, 7, 8]),
        1,
      ),
      throwsArgumentError,
    );
    expect(
      () => quantized.topKEuclidean(
        Float32List.fromList([1, 2, 3, 4, 5, 6, 7, double.nan]),
        1,
      ),
      throwsArgumentError,
    );
  });
}
