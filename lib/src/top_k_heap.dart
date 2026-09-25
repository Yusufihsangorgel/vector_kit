part of 'vector_matrix.dart';

/// Bounded binary min-heap that keeps the largest scores seen.
///
/// The root is the worst of the kept scores. A candidate only causes work
/// when it beats the current worst.
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
