import 'dart:math' as math;
import 'dart:typed_data';

import 'simd.dart';

/// A packed, row-major matrix of single-precision vectors with SIMD
/// top-k similarity search.
///
/// Rows are stored back to back in one [Float32List], padded to a
/// multiple of four components so that every row starts on a 16-byte
/// boundary. That lets the search loops read the whole matrix through a
/// single [Float32x4List] view with no per-row alignment checks and no
/// scalar tails. The padding is internal; [rowAt], [toBytes], and
/// [dimension] only ever expose the logical row.
///
/// L2 norms are computed once per row in [add] and cached, so
/// [topKCosine] costs one SIMD dot product per row.
class VectorMatrix {
  /// Creates an empty matrix whose rows all have [dimension] components.
  ///
  /// Throws [ArgumentError] if [dimension] is not positive.
  VectorMatrix(this.dimension) : _strideLanes = (dimension + 3) >> 2 {
    if (dimension < 1) {
      throw ArgumentError.value(dimension, 'dimension', 'must be positive');
    }
    _capacity = _initialCapacity;
    _data = Float32List(_capacity * _stride);
    _lanes = lanesOf(_data);
    _norms = Float64List(_capacity);
  }

  /// Builds a matrix from [rows], converting each row to single
  /// precision.
  ///
  /// The dimension is taken from the first row. Throws [ArgumentError]
  /// if [rows] is empty, if any row has a different length, or if any
  /// component is NaN or infinite after the conversion to single
  /// precision (a double such as `1e300` becomes infinite and is
  /// rejected).
  factory VectorMatrix.fromRows(List<List<double>> rows) {
    if (rows.isEmpty) {
      throw ArgumentError.value(
        rows,
        'rows',
        'must not be empty; the dimension cannot be inferred',
      );
    }
    final matrix = VectorMatrix(rows.first.length)
      .._ensureCapacity(rows.length);
    for (final row in rows) {
      if (row.length != matrix.dimension) {
        throw ArgumentError.value(
          row,
          'rows',
          'row length ${row.length} does not match the first row '
              '(${matrix.dimension})',
        );
      }
      matrix.add(Float32List.fromList(row));
    }
    return matrix;
  }

  /// Restores a matrix written by [toBytes].
  ///
  /// Throws [FormatException] if [bytes] does not start with the `VKT1`
  /// magic, declares a zero dimension, has a length that does not match
  /// the declared row count, or encodes a NaN or infinite component.
  factory VectorMatrix.fromBytes(Uint8List bytes) {
    if (bytes.length < _headerLength) {
      throw FormatException(
        'too short for a VKT1 header: ${bytes.length} bytes',
      );
    }
    for (var i = 0; i < 4; i++) {
      if (bytes[i] != _magic.codeUnitAt(i)) {
        throw FormatException('bad magic; expected "VKT1"');
      }
    }
    final data = ByteData.sublistView(bytes);
    final dimension = data.getUint32(4, Endian.little);
    final count = data.getUint32(8, Endian.little);
    if (dimension == 0) {
      throw FormatException('dimension must be positive');
    }
    final expected = _headerLength + 4 * dimension * count;
    if (bytes.length != expected) {
      throw FormatException(
        'expected $expected bytes for $count rows of dimension $dimension, '
        'got ${bytes.length}',
      );
    }
    final matrix = VectorMatrix(dimension).._ensureCapacity(count);
    var offset = _headerLength;
    for (var r = 0; r < count; r++) {
      final base = r * matrix._stride;
      for (var c = 0; c < dimension; c++) {
        matrix._data[base + c] = data.getFloat32(offset, Endian.little);
        offset += 4;
      }
    }
    matrix._count = count;
    for (var r = 0; r < count; r++) {
      final laneBase = r * matrix._strideLanes;
      final n2 = dotLanes(
        matrix._lanes,
        laneBase,
        matrix._lanes,
        laneBase,
        matrix._strideLanes,
      );
      if (!n2.isFinite) {
        final bad = firstNonFinite(matrix.rowAt(r));
        if (bad >= 0) {
          throw FormatException('non-finite component at row $r, index $bad');
        }
        throw FormatException('row $r squared norm overflows single precision');
      }
      matrix._norms[r] = math.sqrt(n2);
    }
    return matrix;
  }

  static const int _initialCapacity = 8;
  static const int _headerLength = 12;
  static const String _magic = 'VKT1';

  /// Number of components in every row.
  final int dimension;

  final int _strideLanes;
  late Float32List _data;
  late Float32x4List _lanes;
  late Float64List _norms;
  int _capacity = 0;
  int _count = 0;

  /// Number of rows added so far.
  int get rowCount => _count;

  int get _stride => _strideLanes << 2;

  /// Appends a copy of [row] and caches its L2 norm for [topKCosine].
  ///
  /// Throws [ArgumentError] if [row] does not have [dimension]
  /// components, contains a NaN or infinite component, or has a squared
  /// norm that overflows single precision. On error the matrix is left
  /// unchanged.
  void add(Float32List row) {
    if (row.length != dimension) {
      throw ArgumentError.value(
        row,
        'row',
        'length ${row.length} does not match dimension $dimension',
      );
    }
    _ensureCapacity(_count + 1);
    final base = _count * _stride;
    _data.setRange(base, base + dimension, row);
    final laneBase = _count * _strideLanes;
    // The squared norm doubles as validation: squares cannot cancel, so
    // it is finite exactly when every component is finite and no square
    // overflows single precision.
    final n2 = dotLanes(_lanes, laneBase, _lanes, laneBase, _strideLanes);
    if (!n2.isFinite) {
      final stored = Float32List.sublistView(_data, base, base + dimension);
      final bad = firstNonFinite(stored);
      if (bad >= 0) {
        throw ArgumentError.value(
          row[bad],
          'row',
          'component $bad is not finite',
        );
      }
      throw ArgumentError.value(
        row,
        'row',
        'squared norm overflows single precision',
      );
    }
    _norms[_count] = math.sqrt(n2);
    _count++;
  }

  /// The row at [index] as a live view into the matrix storage, not a
  /// copy.
  ///
  /// Reading through the view is cheap and always reflects the current
  /// storage. Treat the view as read-only: writing through it changes
  /// the stored row but does not update the norm cached for
  /// [topKCosine], so cosine scores for a mutated row become stale.
  ///
  /// Throws [RangeError] if [index] is not in `[0, rowCount)`.
  Float32List rowAt(int index) {
    RangeError.checkValidIndex(index, this, 'index', _count);
    final base = index * _stride;
    return Float32List.sublistView(_data, base, base + dimension);
  }

  /// The [k] rows most similar to [query] by cosine similarity, best
  /// first. Scores are clamped to `[-1, 1]`, matching the top-level
  /// `cosineSimilarity` function.
  ///
  /// Each row costs one SIMD dot product because row norms are cached by
  /// [add]. Rows with zero norm have no cosine score and are skipped, so
  /// the result can hold fewer than [k] entries even when `rowCount >=
  /// k`. At most [rowCount] entries are returned. The order of rows with
  /// exactly equal scores is unspecified.
  ///
  /// Throws [ArgumentError] if [k] is not positive, if [query] does not
  /// have [dimension] components, contains a NaN or infinite component,
  /// is a zero vector, or has a squared norm that overflows single
  /// precision.
  List<(int index, double score)> topKCosine(Float32List query, int k) {
    _checkK(k);
    final lanes = _prepareQuery(query);
    final qn2 = dotLanes(lanes, 0, lanes, 0, _strideLanes);
    if (qn2 == 0) {
      throw ArgumentError.value(
        query,
        'query',
        'is a zero vector; cosine similarity is undefined',
      );
    }
    if (!qn2.isFinite) {
      throw ArgumentError.value(
        query,
        'query',
        'squared norm overflows single precision',
      );
    }
    final qNorm = math.sqrt(qn2);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      final norm = _norms[r];
      if (norm == 0) continue;
      var score =
          dotLanes(lanes, 0, _lanes, r * _strideLanes, _strideLanes) /
          (qNorm * norm);
      if (score > 1) {
        score = 1;
      } else if (score < -1) {
        score = -1;
      }
      heap.offer(r, score);
    }
    return heap.drainDescending();
  }

  /// The [k] rows with the largest dot product against [query], best
  /// first.
  ///
  /// Unlike [topKCosine] a zero query is allowed; every score is then
  /// zero. At most [rowCount] entries are returned. The order of rows
  /// with exactly equal scores is unspecified.
  ///
  /// Throws [ArgumentError] if [k] is not positive or if [query] does
  /// not have [dimension] components or contains a NaN or infinite
  /// component.
  List<(int index, double score)> topKDot(Float32List query, int k) {
    _checkK(k);
    final lanes = _prepareQuery(query);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      heap.offer(r, dotLanes(lanes, 0, _lanes, r * _strideLanes, _strideLanes));
    }
    return heap.drainDescending();
  }

  /// The [k] rows nearest to [query] by Euclidean distance, nearest
  /// first. The score is the distance itself, so smaller is better.
  ///
  /// At most [rowCount] entries are returned. The order of rows with
  /// exactly equal distances is unspecified.
  ///
  /// Throws [ArgumentError] if [k] is not positive or if [query] does
  /// not have [dimension] components or contains a NaN or infinite
  /// component.
  List<(int index, double score)> topKEuclidean(Float32List query, int k) {
    _checkK(k);
    final lanes = _prepareQuery(query);
    final heap = _TopKHeap(math.min(k, _count));
    for (var r = 0; r < _count; r++) {
      // Negated squared distance turns "smallest distance" into "largest
      // score", so the same keep-the-largest heap works for all three
      // searches. The square root is taken only for the k survivors.
      heap.offer(
        r,
        -squaredDistanceLanes(lanes, 0, _lanes, r * _strideLanes, _strideLanes),
      );
    }
    final best = heap.drainDescending();
    return [
      for (final (index, negSquared) in best) (index, math.sqrt(-negSquared)),
    ];
  }

  /// Serializes the matrix without the internal padding.
  ///
  /// Layout: the ASCII bytes `VKT1`, the dimension and the row count as
  /// little-endian uint32, then `rowCount * dimension` float32
  /// components in row-major order, little-endian.
  Uint8List toBytes() {
    final bytes = Uint8List(_headerLength + 4 * dimension * _count);
    final data = ByteData.sublistView(bytes);
    bytes.setRange(0, 4, _magic.codeUnits);
    data.setUint32(4, dimension, Endian.little);
    data.setUint32(8, _count, Endian.little);
    var offset = _headerLength;
    for (var r = 0; r < _count; r++) {
      final base = r * _stride;
      for (var c = 0; c < dimension; c++) {
        data.setFloat32(offset, _data[base + c], Endian.little);
        offset += 4;
      }
    }
    return bytes;
  }

  void _checkK(int k) {
    if (k < 1) {
      throw ArgumentError.value(k, 'k', 'must be positive');
    }
  }

  /// Validates [query] and copies it into padded, aligned scratch
  /// storage so the search loops can run without tails.
  Float32x4List _prepareQuery(Float32List query) {
    if (query.length != dimension) {
      throw ArgumentError.value(
        query,
        'query',
        'length ${query.length} does not match dimension $dimension',
      );
    }
    final padded = Float32List(_stride);
    padded.setRange(0, dimension, query);
    final bad = firstNonFinite(padded);
    if (bad >= 0) {
      throw ArgumentError.value(
        query[bad],
        'query',
        'component $bad is not finite',
      );
    }
    return lanesOf(padded);
  }

  void _ensureCapacity(int rows) {
    if (rows <= _capacity) return;
    var capacity = _capacity;
    while (capacity < rows) {
      capacity *= 2;
    }
    final data = Float32List(capacity * _stride);
    data.setRange(0, _count * _stride, _data);
    _data = data;
    _lanes = lanesOf(_data);
    final norms = Float64List(capacity);
    norms.setRange(0, _count, _norms);
    _norms = norms;
    _capacity = capacity;
  }
}

/// Bounded binary min-heap that keeps the largest scores seen.
///
/// The root is the worst of the kept scores, so a candidate only causes
/// work when it beats the current worst.
class _TopKHeap {
  _TopKHeap(this.capacity)
    : _scores = Float64List(capacity),
      _indices = List<int>.filled(capacity, 0);

  final int capacity;
  final Float64List _scores;
  final List<int> _indices;
  int _size = 0;

  void offer(int index, double score) {
    if (_size < capacity) {
      var i = _size++;
      _scores[i] = score;
      _indices[i] = index;
      while (i > 0) {
        final parent = (i - 1) >> 1;
        if (_scores[parent] <= _scores[i]) break;
        _swap(i, parent);
        i = parent;
      }
    } else if (capacity > 0 && score > _scores[0]) {
      _scores[0] = score;
      _indices[0] = index;
      _siftDown();
    }
  }

  /// Removes everything from the heap, best score first.
  List<(int, double)> drainDescending() {
    final result = List<(int, double)>.filled(_size, (0, 0.0));
    for (var i = _size - 1; i >= 0; i--) {
      result[i] = (_indices[0], _scores[0]);
      _size--;
      if (_size > 0) {
        _scores[0] = _scores[_size];
        _indices[0] = _indices[_size];
        _siftDown();
      }
    }
    return result;
  }

  void _siftDown() {
    var i = 0;
    while (true) {
      final left = 2 * i + 1;
      if (left >= _size) break;
      final right = left + 1;
      var smallest = left;
      if (right < _size && _scores[right] < _scores[left]) {
        smallest = right;
      }
      if (_scores[i] <= _scores[smallest]) break;
      _swap(i, smallest);
      i = smallest;
    }
  }

  void _swap(int a, int b) {
    final score = _scores[a];
    _scores[a] = _scores[b];
    _scores[b] = score;
    final index = _indices[a];
    _indices[a] = _indices[b];
    _indices[b] = index;
  }
}
