// Backing rag_kit's retrieval with vector_kit's packed cosine search.
//
//   dart run example/with_rag_kit.dart
//
// VectorKitStore is a VectorStore, so Retriever cannot tell it from
// InMemoryVectorStore. The class lives in vector_kit_store.dart; this
// file is the runnable walk-through.
import 'package:rag_kit/rag_kit.dart';

import 'vector_kit_store.dart';

Future<void> main() async {
  final store = VectorKitStore();
  final retriever = Retriever(
    embedder: _embed,
    store: store,
    chunker: Chunker.paragraphs(),
  );

  await retriever.addText(
    'A cat sleeping on a keyboard.\n\n'
    'A dog running on a beach.\n\n'
    'A build server compiling Dart.',
    sourceId: 'notes',
  );

  print('stored: ${await store.count()}');

  for (final hit in await retriever.retrieve('cat on a keyboard', topK: 2)) {
    print('  ${hit.score.toStringAsFixed(3)}  ${hit.document.text}');
  }

  // The filter runs against the document, so metadata-scoped retrieval
  // works the same way it does with InMemoryVectorStore.
  final filtered = await retriever.retrieve(
    'compiling Dart',
    topK: 2,
    where: (d) => d.id != 'notes#2',
  );
  print('excluding notes#2: ${filtered.map((h) => h.document.id).toList()}');
}

// Stand-in embeddings. A real one comes from a model; three dimensions
// are enough to see the ranking.
Future<List<List<double>>> _embed(List<String> texts) async {
  return [
    for (final text in texts)
      [
        text.contains('cat') ? 0.9 : 0.1,
        text.contains('dog') ? 0.9 : 0.1,
        text.contains('Dart') ? 0.9 : 0.0,
      ],
  ];
}
