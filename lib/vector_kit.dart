/// SIMD-accelerated vector math for embeddings.
///
/// All operations work on `Float32List`. On the Dart VM the inner loops
/// use `Float32x4` lanes, with scalar handling for the last `length % 4`
/// components; everywhere else they are plain scalar loops, because off
/// the VM `Float32x4` is emulated and the emulation is slower than not
/// using it. `VectorMatrix` packs rows into one padded buffer and caches
/// row norms either way, so top-k cosine search costs one dot product
/// per row.
///
/// The two kernels round differently: the VM accumulates in single
/// precision and the scalar path in double. Scores therefore differ
/// across platforms by around 5e-9, while the rows returned and their
/// order do not. See `doc/web-performance.md`.
///
/// Every entry point validates its input eagerly: length mismatches,
/// NaN or infinite components, and zero vectors where the operation is
/// undefined all throw `ArgumentError` instead of propagating NaN into
/// scores.
library;

export 'src/ops.dart'
    show cosineSimilarity, dot, euclideanDistance, normalized, normalizeInPlace;
export 'src/vector_matrix.dart' show QuantizedMatrix, VectorMatrix;
