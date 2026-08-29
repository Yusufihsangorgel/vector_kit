import 'dart:typed_data';

import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

import '../example/vector_kit_store.dart';

Document doc(String id, List<double> embedding, {Map<String, Object?>? meta}) =>
    Document(
      id: id,
      text: 'text of $id',
      embedding: embedding,
      metadata: meta ?? const {},
    );

Future<void> _fill(VectorStore store, List<Document> documents) async {
  await store.upsert(documents);
}

Future<List<String>> _ids(
  VectorStore store,
  List<double> query, {
  int topK = 5,
  double? minScore,
  bool Function(Document document)? where,
}) async {
  final results = await store.search(
    query,
    topK: topK,
    minScore: minScore,
    where: where,
  );
  return [for (final r in results) r.document.id];
}

void main() {
  group('same ranking as InMemoryVectorStore', () {
    Future<void> expectSame(
      List<Document> documents,
      List<double> query, {
      int topK = 5,
      double? minScore,
      bool Function(Document document)? where,
    }) async {
      final memory = InMemoryVectorStore();
      final packed = VectorKitStore();
      await _fill(memory, documents);
      await _fill(packed, documents);
      final expected = await memory.search(
        query,
        topK: topK,
        minScore: minScore,
        where: where,
      );
      final got = await packed.search(
        query,
        topK: topK,
        minScore: minScore,
        where: where,
      );
      expect(
        [for (final r in got) r.document.id],
        [for (final r in expected) r.document.id],
      );
      expect(got.length, expected.length);
      for (var i = 0; i < expected.length; i++) {
        expect(
          got[i].score,
          closeTo(expected[i].score, 1e-5),
          reason: 'rank $i',
        );
      }
    }

    test('top-k descending score order', () async {
      await expectSame(
        [
          doc('orthogonal', [0, 1]),
          doc('exact', [1, 0]),
          doc('diagonal', [1, 1]),
          doc('close', [2, 1]),
          doc('opposite', [-1, 0]),
        ],
        [1, 0],
        topK: 3,
      );
    });

    test('equal scores keep insertion order', () async {
      await expectSame(
        [
          doc('first', [1, 0]),
          doc('second', [2, 0]),
          doc('third', [3, 0]),
        ],
        [1, 0],
        topK: 2,
      );
    });

    test('ties past the k-cut keep the earlier rows', () async {
      await expectSame(
        [
          doc('a', [1, 0]),
          doc('b', [1, 0]),
          doc('c', [1, 0]),
          doc('d', [1, 0]),
        ],
        [1, 0],
        topK: 2,
      );
    });

    test(
      'zero-vector documents score zero and take their insertion slot',
      () async {
        await expectSame(
          [
            doc('high', [1, 0]),
            doc('zero', [0, 0]),
            doc('opposite', [-1, 0]),
          ],
          [1, 0],
          topK: 3,
        );
      },
    );

    test(
      'a zero-vector query scores every document zero, insertion order',
      () async {
        await expectSame(
          [
            doc('a', [1, 0]),
            doc('b', [0, 1]),
            doc('c', [1, 1]),
          ],
          [0, 0],
          topK: 2,
        );
      },
    );

    test('where is applied before the k-cut, not after', () async {
      // Unfiltered top-1 is the Turkish row. Filtering to English first
      // must return the English row, not an empty list.
      await expectSame(
        [
          doc('tr-1', [1, 0], meta: {'lang': 'tr'}),
          doc('en-1', [0.2, 1], meta: {'lang': 'en'}),
          doc('en-2', [0.1, 1], meta: {'lang': 'en'}),
        ],
        [1, 0],
        topK: 1,
        where: (d) => d.metadata['lang'] == 'en',
      );
    });

    test('where combines with topK', () async {
      await expectSame(
        [
          for (var i = 0; i < 10; i++)
            doc('d$i', [1, i / 10], meta: {'even': i.isEven}),
        ],
        [1, 0],
        topK: 2,
        where: (d) => d.metadata['even'] == true,
      );
    });

    test(
      'filtered zeros keep insertion order among the rows that pass',
      () async {
        await expectSame(
          [
            doc('a', [1, 0], meta: {'keep': true}),
            doc('zero-out', [0, 0], meta: {'keep': false}),
            doc('zero-in', [0, 0], meta: {'keep': true}),
            doc('opposite', [-1, 0], meta: {'keep': true}),
          ],
          [1, 0],
          topK: 3,
          where: (d) => d.metadata['keep'] == true,
        );
      },
    );

    test('minScore drops results below the threshold', () async {
      await expectSame(
        [
          doc('exact', [1, 0]),
          doc('mid', [1, 1]),
          doc('zero', [0, 1]),
        ],
        [1, 0],
        minScore: 0.5,
      );
    });

    test('minScore keeps results exactly at the threshold', () async {
      await expectSame(
        [
          doc('exact', [2, 0]),
          doc('zero', [0, 1]),
        ],
        [1, 0],
        minScore: 1.0,
      );
    });

    test('topK larger than the store returns everything, same order', () async {
      await expectSame(
        [
          doc('a', [1, 0]),
          doc('b', [0, 1]),
        ],
        [1, 0],
        topK: 50,
      );
    });

    test('re-upsert of an id keeps its insertion position on ties', () async {
      final documents = [
        doc('first', [1, 0]),
        doc('second', [2, 0]),
        doc('third', [3, 0]),
      ];
      final memory = InMemoryVectorStore();
      final packed = VectorKitStore();
      await _fill(memory, documents);
      await _fill(packed, documents);
      final replacement = Document(
        id: 'second',
        text: 'updated',
        embedding: [4, 0],
      );
      await memory.upsert([replacement]);
      await packed.upsert([replacement]);
      expect(
        await _ids(packed, [1, 0], topK: 3),
        await _ids(memory, [1, 0], topK: 3),
      );
      expect(await packed.count(), await memory.count());
      final hits = await packed.search([1, 0], topK: 3);
      expect(hits[1].document.text, 'updated');
    });

    test('duplicate ids in one batch keep the last write, one row', () async {
      final batch = [
        doc('a', [1, 0]),
        doc('a', [0, 1]),
      ];
      final memory = InMemoryVectorStore();
      final packed = VectorKitStore();
      await memory.upsert(batch);
      await packed.upsert(batch);
      expect(await packed.count(), await memory.count());
      expect(
        await _ids(packed, [0, 1], topK: 1),
        await _ids(memory, [0, 1], topK: 1),
      );
    });

    test('append after a search does not duplicate rows', () async {
      final memory = InMemoryVectorStore();
      final packed = VectorKitStore();
      await _fill(memory, [
        doc('a', [1, 0]),
      ]);
      await _fill(packed, [
        doc('a', [1, 0]),
      ]);
      await memory.search([1, 0]);
      await packed.search([1, 0]);
      await memory.upsert([
        doc('b', [0, 1]),
      ]);
      await packed.upsert([
        doc('b', [0, 1]),
      ]);
      expect(await packed.count(), 2);
      expect(
        await _ids(packed, [1, 0], topK: 2),
        await _ids(memory, [1, 0], topK: 2),
      );
    });

    test('removeWhere then search agrees', () async {
      final documents = [
        doc('a', [1, 0], meta: {'kind': 'old'}),
        doc('b', [0, 1], meta: {'kind': 'new'}),
        doc('c', [1, 1], meta: {'kind': 'old'}),
      ];
      final memory = InMemoryVectorStore();
      final packed = VectorKitStore();
      await _fill(memory, documents);
      await _fill(packed, documents);
      expect(
        await packed.removeWhere((d) => d.metadata['kind'] == 'old'),
        await memory.removeWhere((d) => d.metadata['kind'] == 'old'),
      );
      expect(await _ids(packed, [0, 1]), await _ids(memory, [0, 1]));
    });
  });

  group('VectorKitStore contract', () {
    test('hand-computed cosine', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [4, 5, 6]),
      ]);
      final results = await store.search([1, 2, 3], topK: 1);
      expect(results.single.score, closeTo(0.9746318461970762, 1e-6));
    });

    test('orthogonal vectors score zero', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [0, 1]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, closeTo(0, 1e-9));
    });

    test('opposite vectors score minus one', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [-1, 0]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, closeTo(-1, 1e-6));
    });

    test('same direction scores one regardless of magnitude', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [2, 0]),
      ]);
      final results = await store.search([0.5, 0]);
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('a zero-vector document scores zero, not NaN', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('zero', [0, 0]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, 0);
      expect(results.single.score.isNaN, isFalse);
    });

    test('empty store returns an empty list for any query', () async {
      final store = VectorKitStore();
      expect(await store.search([1, 2, 3]), isEmpty);
    });

    test('topK below one throws', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.search([1, 0], topK: 0),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('query dimension mismatch throws with a clear message', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.search([1, 0, 0]),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            allOf(contains('3 dimensions'), contains('2-dimensional')),
          ),
        ),
      );
    });

    test('upsert with the same id replaces the document', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await store.upsert([
        Document(id: 'a', text: 'updated', embedding: [0, 1]),
      ]);
      expect(await store.count(), 1);
      final results = await store.search([0, 1]);
      expect(results.single.document.text, 'updated');
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('dimension mismatch on upsert throws', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.upsert([
          doc('b', [1, 0, 0]),
        ]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('a mismatched batch is rejected without partial writes', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.upsert([
          doc('b', [0, 1]),
          doc('c', [1, 2, 3]),
        ]),
        throwsA(isA<ArgumentError>()),
      );
      expect(await store.count(), 1);
    });

    test('an empty embedding throws', () async {
      final store = VectorKitStore();
      await expectLater(
        store.upsert([doc('a', [])]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('embeddings are stored as float32', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [0.1, 0.2]),
      ]);
      final results = await store.search([1, 1]);
      final stored = results.single.document.embedding;
      final expected = Float32List.fromList([0.1, 0.2]);
      expect(stored[0], expected[0]);
      expect(stored[1], expected[1]);
    });

    test('metadata is copied at upsert time, not aliased', () async {
      final store = VectorKitStore();
      final original = <String, Object?>{'status': 'draft'};
      await store.upsert([
        doc('a', [1, 0], meta: original),
      ]);
      original['status'] = 'published';
      original['secret'] = 'leaked';
      final results = await store.search([1, 0]);
      expect(results.single.document.metadata, {'status': 'draft'});
    });

    test('metadata returned from search is unmodifiable', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0], meta: {'status': 'draft'}),
      ]);
      final results = await store.search([1, 0]);
      expect(
        () => results.single.document.metadata['status'] = 'hacked',
        throwsUnsupportedError,
      );
      final again = await store.search([1, 0]);
      expect(again.single.document.metadata, {'status': 'draft'});
    });

    test('count, clear, and dimension lifecycle', () async {
      final store = VectorKitStore();
      expect(store.dimension, isNull);
      await store.upsert([
        doc('a', [1, 0]),
        doc('b', [0, 1]),
      ]);
      expect(await store.count(), 2);
      expect(store.dimension, 2);
      await store.clear();
      expect(await store.count(), 0);
      expect(store.dimension, isNull);
      await store.upsert([
        doc('c', [1, 2, 3]),
      ]);
      expect(store.dimension, 3);
    });

    test('removeWhere removes matches and reports the count', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0], meta: {'kind': 'old'}),
        doc('b', [0, 1], meta: {'kind': 'new'}),
        doc('c', [1, 1], meta: {'kind': 'old'}),
      ]);
      final removed = await store.removeWhere(
        (d) => d.metadata['kind'] == 'old',
      );
      expect(removed, 2);
      expect(await store.count(), 1);
      final results = await store.search([0, 1]);
      expect(results.single.document.id, 'b');
    });

    test('removing every document resets the dimension', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await store.removeWhere((_) => true);
      expect(store.dimension, isNull);
      await store.upsert([
        doc('b', [1, 2, 3]),
      ]);
      expect(store.dimension, 3);
    });

    test('rejects NaN, infinity, and float32-overflowing components', () async {
      final store = VectorKitStore();
      for (final bad in [double.nan, double.infinity, -double.infinity, 1e39]) {
        await expectLater(
          store.upsert([
            doc('bad', [bad, 0]),
          ]),
          throwsArgumentError,
          reason: 'component $bad must be rejected',
        );
      }
      expect(await store.count(), 0);
    });

    test('rejects non-finite query components', () async {
      final store = VectorKitStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      for (final bad in [double.nan, double.infinity, 1e39]) {
        await expectLater(store.search([bad, 0]), throwsArgumentError);
      }
    });

    test('a rejected batch writes nothing', () async {
      final store = VectorKitStore();
      await expectLater(
        store.upsert([
          doc('ok', [1, 0]),
          doc('bad', [double.nan, 0]),
        ]),
        throwsArgumentError,
      );
      expect(await store.count(), 0);
    });

    test(
      'where that matches nothing returns empty, after query checks',
      () async {
        final store = VectorKitStore();
        await store.upsert([
          doc('a', [1, 0]),
        ]);
        expect(await store.search([1, 0], where: (_) => false), isEmpty);
        await expectLater(
          store.search([1, 0, 0], where: (_) => false),
          throwsA(isA<ArgumentError>()),
        );
      },
    );
  });

  group('Retriever over VectorKitStore', () {
    test('addText then retrieve ranks the matching chunk first', () async {
      final store = VectorKitStore();
      final retriever = Retriever(
        embedder: (texts) async => [
          for (final text in texts)
            [
              text.contains('cat') ? 1.0 : 0.0,
              text.contains('dog') ? 1.0 : 0.0,
            ],
        ],
        store: store,
        chunker: Chunker.paragraphs(),
      );
      await retriever.addText(
        'the cat sleeps\n\nthe dog runs',
        sourceId: 'pets',
      );
      expect(await store.count(), 2);
      final hits = await retriever.retrieve('cat', topK: 1);
      expect(hits.single.document.id, 'pets#0');
    });
  });
}
