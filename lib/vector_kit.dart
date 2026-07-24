/// SIMD-accelerated vector math for embeddings.
///
/// All operations work on `Float32List` and use `Float32x4` lanes in
/// their inner loops, with scalar handling for the last `length % 4`
/// components. `VectorMatrix` packs rows into one padded buffer and
/// caches row norms, so top-k cosine search costs one SIMD dot product
/// per row.
///
/// Every entry point validates its input eagerly: length mismatches,
/// NaN or infinite components, and zero vectors where the operation is
/// undefined all throw `ArgumentError` instead of propagating NaN into
/// scores.
library;

export 'src/ops.dart'
    show cosineSimilarity, dot, euclideanDistance, normalized, normalizeInPlace;
export 'src/vector_matrix.dart' show QuantizedMatrix, VectorMatrix;
