part of 'vector_matrix.dart';

/// A [VectorMatrix] stored as 8-bit integers, for when the vectors no longer
/// fit comfortably in memory.
///
/// Each row is scaled so its largest component maps to 127 and stored as one
/// byte per dimension, next to the scale that undoes it. For 768-dimension
/// embeddings that is 768 bytes a row instead of 3072, so a corpus that took
/// 300 MB takes about 75 MB, which is the difference between holding an index
/// on a phone and not.
///
/// The trade is accuracy and speed, in that order. Scores come back close to
/// the float32 ones but not equal, so measure recall on your own vectors
/// before trusting it: [QuantizedMatrix.from] keeps the original matrix
/// untouched so the two can be compared directly. Search is also slower here,
/// not faster, because the byte rows cannot go through the same SIMD path the
/// float rows do; this buys memory, not throughput.
final class QuantizedMatrix {
  QuantizedMatrix._(
    this.dimension,
    this._count,
    this._data,
    this._scales,
    this._norms,
  );

  /// Quantizes every row of [source].
  ///
  /// [source] is not modified and stays usable, which is what lets you check
  /// recall against it.
  factory QuantizedMatrix.from(VectorMatrix source) {
    final dimension = source.dimension;
    final count = source.rowCount;
    final data = Int8List(count * dimension);
    final scales = Float64List(count);
    final norms = Float64List(count);
    for (var r = 0; r < count; r++) {
      final row = source.rowAt(r);
      var maxAbs = 0.0;
      for (var i = 0; i < dimension; i++) {
        final a = row[i].abs();
        if (a > maxAbs) maxAbs = a;
      }
      // An all-zero row has no scale that means anything; leave it zero and
      // let the search skip it the way the float path skips a zero norm.
      final scale = maxAbs == 0 ? 0.0 : maxAbs / 127.0;
      scales[r] = scale;
      final base = r * dimension;
      var sum2 = 0.0;
      for (var i = 0; i < dimension; i++) {
        final q = scale == 0 ? 0 : (row[i] / scale).round().clamp(-127, 127);
        data[base + i] = q;
        // Norm of what is actually stored, so cosine is exact with respect to
        // the rounded vector rather than to the original it came from.
        final dequantized = q * scale;
        sum2 += dequantized * dequantized;
      }
      norms[r] = math.sqrt(sum2);
    }
    return QuantizedMatrix._(dimension, count, data, scales, norms);
  }

  /// Components per row.
  final int dimension;

  final int _count;
  final Int8List _data;
  final Float64List _scales;
  final Float64List _norms;

  /// The number of rows.
  int get rowCount => _count;

  /// Bytes held by the quantized rows and their scales and norms.
  ///
  /// Compare with `matrix.rowCount * matrix.dimension * 4` for the float32
  /// storage it replaces.
  int get byteSize =>
      _data.lengthInBytes + _scales.lengthInBytes + _norms.lengthInBytes;

  /// The [k] rows most similar to [query] by cosine, best first.
  ///
  /// [query] stays in full precision; only the stored rows are quantized.
  /// Throws [ArgumentError] if [k] is below 1, if [query] has the wrong
  /// length, contains a NaN or infinite component, or is a zero vector.
  List<(int index, double score)> topKCosine(List<double> query, int k) {
    _check(query, k);
    var qn2 = 0.0;
    for (var i = 0; i < dimension; i++) {
      qn2 += query[i] * query[i];
    }
    if (qn2 == 0) {
      throw ArgumentError.value(
        query,
        'query',
        'is a zero vector; cosine similarity is undefined',
      );
    }
    final qNorm = math.sqrt(qn2);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      final norm = _norms[r];
      if (norm == 0) continue;
      final scale = _scales[r];
      final base = r * dimension;
      var dot = 0.0;
      for (var i = 0; i < dimension; i++) {
        dot += query[i] * _data[base + i];
      }
      heap.offer(r, dot * scale / (qNorm * norm));
    }
    return heap.drainDescending();
  }

  /// The [k] rows with the largest dot product against [query], best first.
  ///
  /// Throws [ArgumentError] if [k] is below 1, if [query] has the wrong
  /// length, or if [query] contains a NaN or infinite component.
  List<(int index, double score)> topKDot(List<double> query, int k) {
    _check(query, k);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      final scale = _scales[r];
      final base = r * dimension;
      var dot = 0.0;
      for (var i = 0; i < dimension; i++) {
        dot += query[i] * _data[base + i];
      }
      heap.offer(r, dot * scale);
    }
    return heap.drainDescending();
  }

  /// The [k] rows nearest to [query] by Euclidean distance, nearest first.
  /// The score is the distance itself, so smaller is better.
  ///
  /// Distances are measured against the stored rows, which are the rounded
  /// ones, the same way [topKCosine] scores against the norm of what is
  /// stored rather than the norm of the row it came from.
  ///
  /// Two things differ from [topKCosine], both because a distance is defined
  /// where an angle is not. A zero row is scored instead of skipped, since
  /// the origin is a real point and can be the nearest one, so the result
  /// always holds `min(k, rowCount)` entries. A zero query is allowed, and
  /// then each row is scored by its own norm.
  ///
  /// The order of rows with exactly equal distances is unspecified.
  ///
  /// Throws [ArgumentError] if [k] is below 1, if [query] has the wrong
  /// length, or if [query] contains a NaN or infinite component.
  List<(int index, double score)> topKEuclidean(List<double> query, int k) {
    _check(query, k);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      final scale = _scales[r];
      final base = r * dimension;
      // Accumulate the differences directly rather than expanding to
      // |q|^2 - 2<q,x> + |x|^2, which would reuse the dot loop above and the
      // norm already cached per row. The expansion reaches a small distance
      // by subtracting large numbers and loses exactly the digits a
      // nearest-neighbour search is reading: on a 768-component row it
      // reports 1.00012e-3 for a separation of 1e-3. See the near-miss test.
      var sum2 = 0.0;
      for (var i = 0; i < dimension; i++) {
        final d = query[i] - _data[base + i] * scale;
        sum2 += d * d;
      }
      // Negated so the keep-the-largest heap shared with the other two
      // searches keeps the nearest rows rather than the farthest. A sum of
      // squares cannot be negative, so the root below needs no clamp.
      heap.offer(r, -sum2);
    }
    return [
      for (final (index, negSquared) in heap.drainDescending())
        (index, math.sqrt(-negSquared)),
    ];
  }

  void _check(List<double> query, int k) {
    if (k < 1) {
      throw ArgumentError.value(k, 'k', 'must be at least 1');
    }
    if (query.length != dimension) {
      throw ArgumentError.value(
        query.length,
        'query',
        'must have $dimension components',
      );
    }
    for (var i = 0; i < query.length; i++) {
      if (!query[i].isFinite) {
        throw ArgumentError.value(
          query[i],
          'query',
          'component $i is not finite',
        );
      }
    }
  }
}
