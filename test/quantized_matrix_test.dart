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

  test('bad arguments are rejected', () {
    final quantized = QuantizedMatrix.from(_corpus(rows: 4, dimension: 8));
    expect(() => quantized.topKCosine(Float32List(8), 0), throwsArgumentError);
    expect(() => quantized.topKCosine(Float32List(7), 1), throwsArgumentError);
    // An all-zero query has no direction to compare against.
    expect(() => quantized.topKCosine(Float32List(8), 1), throwsArgumentError);
  });
}
