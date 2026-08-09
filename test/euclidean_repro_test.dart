// TEMPORARY reproduction file. Delete after capturing the failure.
import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

void main() {
  test('quantizing keeps all three searches', () {
    final matrix = VectorMatrix.fromRows([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0],
    ]);
    final query = <double>[1.0, 0.0, 0.0, 0.0];
    final quantized = QuantizedMatrix.from(matrix);

    expect(quantized.topKCosine(query, 2).length, 2);
    expect(quantized.topKDot(query, 2).length, 2);
    expect(quantized.topKEuclidean(query, 2).length, 2);
  });
}
