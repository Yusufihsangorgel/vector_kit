import 'dart:math' as math;
import 'dart:typed_data';

import 'simd.dart';

void _checkPair(Float32List a, Float32List b) {
  if (a.isEmpty) {
    throw ArgumentError.value(a, 'a', 'must not be empty');
  }
  if (b.length != a.length) {
    throw ArgumentError.value(
      b,
      'b',
      'length ${b.length} does not match a (length ${a.length})',
    );
  }
}

/// Throws the [ArgumentError] appropriate for a non-finite accumulation
/// over [aligned]: either the exact non-finite component, or overflow
/// when every component turns out to be finite.
Never _throwNonFinite(Float32List original, Float32List aligned, String name) {
  final bad = firstNonFinite(aligned);
  if (bad >= 0) {
    throw ArgumentError.value(
      original[bad],
      name,
      'component $bad is not finite',
    );
  }
  throw ArgumentError.value(
    original,
    name,
    'squared norm overflows single precision; scale the input down first',
  );
}

/// Returns the dot product of [a] and [b].
///
/// The inner loop processes four components per step with [Float32x4]
/// and keeps four independent accumulators; the last `length % 4`
/// components are added with scalar arithmetic. Accumulation is single
/// precision, so the result can differ from an exact double-precision
/// sum by a small amount that grows with vector length.
///
/// The finiteness check costs nothing on the fast path: a NaN or
/// infinite component always drives the accumulated sum non-finite
/// (infinities and NaNs never cancel back to a finite value under
/// multiply-add), so the inputs are only rescanned to locate the
/// offending component when the sum comes back non-finite. If that
/// rescan finds only finite components the accumulation itself
/// overflowed single precision, and the infinite (in rare cancellation
/// cases NaN) sum is returned as is.
///
/// Throws [ArgumentError] if the vectors are empty, have different
/// lengths, or contain a NaN or infinite component.
double dot(Float32List a, Float32List b) {
  _checkPair(a, b);
  final ca = alignedView(a);
  final cb = alignedView(b);
  final sum = dotFull(ca, cb);
  if (sum.isFinite) return sum;
  final badA = firstNonFinite(ca);
  if (badA >= 0) {
    throw ArgumentError.value(a[badA], 'a', 'component $badA is not finite');
  }
  final badB = firstNonFinite(cb);
  if (badB >= 0) {
    throw ArgumentError.value(b[badB], 'b', 'component $badB is not finite');
  }
  return sum;
}

/// Returns the cosine similarity of [a] and [b], clamped to `[-1, 1]`.
///
/// Cosine similarity is the dot product of the two vectors divided by
/// the product of their L2 norms. Single-precision rounding can push the
/// raw ratio slightly past 1 for near-parallel vectors, so the result is
/// clamped.
///
/// Throws [ArgumentError] if the vectors are empty, have different
/// lengths, contain a NaN or infinite component, are zero vectors, or
/// have squared norms that overflow single precision.
double cosineSimilarity(Float32List a, Float32List b) {
  _checkPair(a, b);
  final ca = alignedView(a);
  final cb = alignedView(b);
  // A finite squared norm proves every component finite: squares are
  // non-negative, so a NaN or infinity in the input cannot cancel out
  // of the sum. That makes the norm computation double as validation.
  final na2 = dotFull(ca, ca);
  if (!na2.isFinite) _throwNonFinite(a, ca, 'a');
  if (na2 == 0) {
    throw ArgumentError.value(
      a,
      'a',
      'is a zero vector; cosine similarity is undefined',
    );
  }
  final nb2 = dotFull(cb, cb);
  if (!nb2.isFinite) _throwNonFinite(b, cb, 'b');
  if (nb2 == 0) {
    throw ArgumentError.value(
      b,
      'b',
      'is a zero vector; cosine similarity is undefined',
    );
  }
  final c = dotFull(ca, cb) / (math.sqrt(na2) * math.sqrt(nb2));
  if (c.isNaN) {
    // Reachable only with components near the single-precision limit:
    // both norms finite but the cross products overflow and cancel to
    // NaN inside the accumulator. Refuse rather than return NaN.
    throw ArgumentError(
      'accumulation overflowed single precision; scale the inputs down',
    );
  }
  if (c > 1) return 1;
  if (c < -1) return -1;
  return c;
}

/// Returns the Euclidean (L2) distance between [a] and [b].
///
/// Computed as the square root of the SIMD-accumulated squared
/// distance. The precision and validation notes on [dot] apply here as
/// well: the finiteness of the inputs is proven by a finite squared
/// distance, and the inputs are only rescanned when it is non-finite.
///
/// Throws [ArgumentError] if the vectors are empty, have different
/// lengths, or contain a NaN or infinite component.
double euclideanDistance(Float32List a, Float32List b) {
  _checkPair(a, b);
  final ca = alignedView(a);
  final cb = alignedView(b);
  final sum = squaredDistanceFull(ca, cb);
  if (sum.isFinite) return math.sqrt(sum);
  final badA = firstNonFinite(ca);
  if (badA >= 0) {
    throw ArgumentError.value(a[badA], 'a', 'component $badA is not finite');
  }
  final badB = firstNonFinite(cb);
  if (badB >= 0) {
    throw ArgumentError.value(b[badB], 'b', 'component $badB is not finite');
  }
  return math.sqrt(sum);
}

/// Scales [v] in place so that its L2 norm becomes 1.
///
/// Throws [ArgumentError] if [v] is empty, contains a NaN or infinite
/// component, is a zero vector, or has a squared norm that overflows
/// single precision. On error [v] is left unchanged.
void normalizeInPlace(Float32List v) {
  if (v.isEmpty) {
    throw ArgumentError.value(v, 'v', 'must not be empty');
  }
  final cv = alignedView(v);
  final n2 = dotFull(cv, cv);
  if (!n2.isFinite) _throwNonFinite(v, cv, 'v');
  if (n2 == 0) {
    throw ArgumentError.value(
      v,
      'v',
      'is a zero vector; it cannot be normalized',
    );
  }
  scaleInPlace(v, 1.0 / math.sqrt(n2));
}

/// Returns a unit-norm copy of [v]. [v] itself is not modified.
///
/// Throws [ArgumentError] under the same conditions as
/// [normalizeInPlace].
Float32List normalized(Float32List v) {
  final out = Float32List.fromList(v);
  normalizeInPlace(out);
  return out;
}
