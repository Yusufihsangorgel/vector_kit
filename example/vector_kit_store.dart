// A rag_kit VectorStore backed by VectorMatrix.
//
//   dart run example/with_rag_kit.dart
//
// This file is not under lib/: VectorStore, Document, and ScoredChunk live
// in rag_kit, and importing them from the public library would make rag_kit
// a runtime dependency of every vector_kit user. It is a dev dependency
// here, used by this adapter and by test/rag_kit_store_test.dart.
//
// Copy this file into an app that already depends on both packages.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:rag_kit/rag_kit.dart';
import 'package:vector_kit/vector_kit.dart';

/// A [VectorStore] that scores with [VectorMatrix.topKCosine] on the
/// unfiltered path.
///
/// Worth it on the Dart VM from a few thousand chunks up. rag_kit's
/// [InMemoryVectorStore] already caches L2 norms and keeps a k-heap, which
/// is the careful loop in `test/platform_cost_test.dart`, not the naive
/// sort in `bench/break_even.dart`. That loop costs a millisecond between
/// 1,000 and 3,200 rows of 768 dimensions; below that, keep the in-memory
/// store. On the web the packed scan is slower than the loop, so this is
/// not a speedup there.
///
/// [VectorMatrix] is append-only. A replace or a removal rebuilds it.
/// Index once and query many times; a write-heavy store wants a different
/// structure. Documents are kept alongside the packed copy of their
/// embeddings, so the float32 footprint is roughly twice
/// [InMemoryVectorStore] at the same count.
final class VectorKitStore extends VectorStore {
  /// The largest finite float32 value. Finite doubles beyond this would
  /// silently overflow to infinity when stored as float32, so they are
  /// rejected up front together with NaN and infinity.
  static const double _maxFloat32 = 3.4028234663852886e38;

  final Map<String, _Entry> _entries = <String, _Entry>{};
  VectorMatrix? _matrix;
  int? _dimension;
  int _zeroNormCount = 0;

  /// The embedding dimension of the stored documents, or null while the
  /// store is empty.
  ///
  /// Set by the first upsert and reset when the store becomes empty again,
  /// after which documents of a different dimension are accepted.
  int? get dimension => _dimension;

  @override
  Future<void> upsert(List<Document> documents) async {
    if (documents.isEmpty) return;

    var dimension = _dimension;
    for (final document in documents) {
      final length = document.embedding.length;
      if (length == 0) {
        throw ArgumentError(
          'Document "${document.id}" has an empty embedding.',
        );
      }
      if (dimension != null && length != dimension) {
        throw ArgumentError(
          'Document "${document.id}" has $length dimensions, but the store '
          'holds $dimension-dimensional embeddings.',
        );
      }
      dimension ??= length;
      for (var i = 0; i < length; i++) {
        final value = document.embedding[i];
        if (!value.isFinite || value.abs() > _maxFloat32) {
          throw ArgumentError(
            'Document "${document.id}" has an embedding component that is '
            'not representable as a finite float32: $value at index $i. '
            'A NaN or infinite score would corrupt every search ranking.',
          );
        }
      }
    }

    // Replacing an id, or seeing the same id twice in this batch, would
    // leave the append-only matrix with a stale or duplicate row.
    var rebuild = false;
    final seen = <String>{};
    for (final document in documents) {
      if (_entries.containsKey(document.id) || !seen.add(document.id)) {
        rebuild = true;
        break;
      }
    }

    final matrix = rebuild ? null : _matrix;
    if (rebuild) _matrix = null;

    for (final document in documents) {
      final entry = _insert(document);
      matrix?.add(entry.vector);
    }
    _dimension = dimension;
  }

  @override
  Future<List<ScoredChunk>> search(
    List<double> query, {
    int topK = 5,
    double? minScore,
    bool Function(Document document)? where,
  }) async {
    if (topK < 1) {
      throw ArgumentError.value(topK, 'topK', 'must be at least 1');
    }
    if (_entries.isEmpty) return const [];
    final dimension = _dimension!;
    if (query.length != dimension) {
      throw ArgumentError(
        'Query has ${query.length} dimensions, but the store holds '
        '$dimension-dimensional embeddings.',
      );
    }
    for (var i = 0; i < query.length; i++) {
      final value = query[i];
      if (!value.isFinite || value.abs() > _maxFloat32) {
        throw ArgumentError(
          'Query has a component that is not representable as a finite '
          'float32: $value at index $i.',
        );
      }
    }

    // Filter first so excluded documents are not scored.
    final candidates = <_Entry>[];
    final orders = <int>[];
    var order = 0;
    for (final entry in _entries.values) {
      if (where == null || where(entry.document)) {
        candidates.add(entry);
        orders.add(order);
      }
      order++;
    }
    if (candidates.isEmpty) return const [];

    final queryVector = Float32List.fromList(query);
    var querySumSquares = 0.0;
    for (var i = 0; i < queryVector.length; i++) {
      querySumSquares += queryVector[i] * queryVector[i];
    }
    final queryNorm = math.sqrt(querySumSquares);

    if (queryNorm == 0) {
      return _selectZeroScores(candidates, orders, topK, minScore);
    }

    if (candidates.length == _entries.length && _zeroNormCount == 0) {
      return _searchPacked(query, candidates, topK, minScore);
    }
    if (candidates.length == _entries.length) {
      return _searchPackedWithZeros(query, candidates, topK, minScore);
    }
    return _searchPairwise(
      queryVector,
      queryNorm,
      candidates,
      orders,
      topK,
      minScore,
    );
  }

  @override
  Future<int> removeWhere(bool Function(Document document) test) async {
    final before = _entries.length;
    _entries.removeWhere((_, entry) {
      final drop = test(entry.document);
      if (drop && entry.norm == 0) _zeroNormCount--;
      return drop;
    });
    final removed = before - _entries.length;
    if (removed > 0) _matrix = null;
    if (_entries.isEmpty) {
      _dimension = null;
      _zeroNormCount = 0;
    }
    return removed;
  }

  @override
  Future<int> count() async => _entries.length;

  @override
  Future<void> clear() async {
    _entries.clear();
    _matrix = null;
    _dimension = null;
    _zeroNormCount = 0;
  }

  _Entry _insert(Document document) {
    final vector = Float32List.fromList(document.embedding);
    var sumSquares = 0.0;
    for (var i = 0; i < vector.length; i++) {
      final value = vector[i];
      sumSquares += value * value;
    }
    final norm = math.sqrt(sumSquares);
    final previous = _entries[document.id];
    if (previous != null && previous.norm == 0) _zeroNormCount--;
    if (norm == 0) _zeroNormCount++;
    final entry = _Entry(
      document: Document(
        id: document.id,
        text: document.text,
        embedding: vector,
        metadata: Map<String, Object?>.unmodifiable(document.metadata),
      ),
      vector: vector,
      norm: norm,
    );
    _entries[document.id] = entry;
    return entry;
  }

  VectorMatrix _build() {
    final matrix = VectorMatrix(_dimension!);
    for (final entry in _entries.values) {
      matrix.add(entry.vector);
    }
    return matrix;
  }

  List<ScoredChunk> _searchPacked(
    List<double> query,
    List<_Entry> entries,
    int topK,
    double? minScore,
  ) {
    final matrix = _matrix ??= _build();
    final ranked = matrix.topKCosine(query, math.min(topK, matrix.rowCount));
    ranked.sort((a, b) {
      final byScore = b.$2.compareTo(a.$2);
      if (byScore != 0) return byScore;
      return a.$1.compareTo(b.$1);
    });
    final results = <ScoredChunk>[];
    for (final (index, score) in ranked) {
      if (minScore != null && score < minScore) continue;
      results.add(ScoredChunk(document: entries[index].document, score: score));
    }
    return results;
  }

  /// [VectorMatrix.topKCosine] skips zero-norm rows; rag_kit scores them
  /// as 0, so they have to be merged back before the k-cut.
  List<ScoredChunk> _searchPackedWithZeros(
    List<double> query,
    List<_Entry> entries,
    int topK,
    double? minScore,
  ) {
    final matrix = _matrix ??= _build();
    final ranked = matrix.topKCosine(query, matrix.rowCount);
    final scores = List<double>.filled(entries.length, 0.0);
    for (final (index, score) in ranked) {
      scores[index] = score;
    }
    return _select(entries, scores, topK, minScore);
  }

  List<ScoredChunk> _searchPairwise(
    Float32List query,
    double queryNorm,
    List<_Entry> entries,
    List<int> orders,
    int topK,
    double? minScore,
  ) {
    final heap = _HitHeap(topK);
    for (var i = 0; i < entries.length; i++) {
      final score = _score(query, queryNorm, entries[i]);
      if (minScore != null && score < minScore) continue;
      heap.offer(orders[i], score, entries[i].document);
    }
    return heap.drain();
  }

  List<ScoredChunk> _selectZeroScores(
    List<_Entry> entries,
    List<int> orders,
    int topK,
    double? minScore,
  ) {
    if (minScore != null && minScore > 0) return const [];
    final heap = _HitHeap(topK);
    for (var i = 0; i < entries.length; i++) {
      heap.offer(orders[i], 0, entries[i].document);
    }
    return heap.drain();
  }

  List<ScoredChunk> _select(
    List<_Entry> entries,
    List<double> scores,
    int topK,
    double? minScore,
  ) {
    final heap = _HitHeap(topK);
    for (var i = 0; i < entries.length; i++) {
      final score = scores[i];
      if (minScore != null && score < minScore) continue;
      heap.offer(i, score, entries[i].document);
    }
    return heap.drain();
  }

  static double _score(Float32List query, double queryNorm, _Entry entry) {
    if (queryNorm == 0 || entry.norm == 0) return 0;
    var score = dot(query, entry.vector) / (queryNorm * entry.norm);
    if (score > 1) return 1;
    if (score < -1) return -1;
    return score;
  }
}

class _Entry {
  _Entry({required this.document, required this.vector, required this.norm});

  final Document document;
  final Float32List vector;
  final double norm;
}

class _Hit {
  _Hit(this.order, this.score, this.document);

  final int order;
  final double score;
  final Document document;
}

/// Bounded min-heap of the best hits, with insertion order as the
/// tie-break so equal scores match [InMemoryVectorStore].
class _HitHeap {
  _HitHeap(this.capacity);

  final int capacity;
  final List<_Hit> _heap = <_Hit>[];

  static bool _beats(_Hit a, _Hit b) =>
      a.score > b.score || (a.score == b.score && a.order < b.order);

  void offer(int order, double score, Document document) {
    final hit = _Hit(order, score, document);
    if (_heap.length < capacity) {
      _heap.add(hit);
      _siftUp(_heap.length - 1);
    } else if (capacity > 0 && _beats(hit, _heap[0])) {
      _heap[0] = hit;
      _siftDown();
    }
  }

  void _siftUp(int index) {
    var child = index;
    while (child > 0) {
      final parent = (child - 1) >> 1;
      if (!_beats(_heap[parent], _heap[child])) break;
      _swap(parent, child);
      child = parent;
    }
  }

  void _siftDown() {
    final length = _heap.length;
    var parent = 0;
    while (true) {
      var lowest = parent;
      final left = 2 * parent + 1;
      final right = left + 1;
      if (left < length && _beats(_heap[lowest], _heap[left])) {
        lowest = left;
      }
      if (right < length && _beats(_heap[lowest], _heap[right])) {
        lowest = right;
      }
      if (lowest == parent) return;
      _swap(parent, lowest);
      parent = lowest;
    }
  }

  void _swap(int a, int b) {
    final tmp = _heap[a];
    _heap[a] = _heap[b];
    _heap[b] = tmp;
  }

  List<ScoredChunk> drain() {
    _heap.sort((a, b) {
      final byScore = b.score.compareTo(a.score);
      if (byScore != 0) return byScore;
      return a.order.compareTo(b.order);
    });
    return [
      for (final hit in _heap)
        ScoredChunk(document: hit.document, score: hit.score),
    ];
  }
}
