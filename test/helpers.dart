import 'dart:math';
import 'dart:typed_data';

/// A random vector with components uniform in `[-scale, scale)`.
Float32List randomVector(int length, Random rng, {double scale = 1.0}) =>
    Float32List.fromList([
      for (var i = 0; i < length; i++) (rng.nextDouble() * 2 - 1) * scale,
    ]);

/// Double-precision reference dot product over the (already
/// float32-rounded) components of [a] and [b].
double refDot(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

double refNorm(Float32List v) => sqrt(refDot(v, v));

double refCosine(Float32List a, Float32List b) =>
    refDot(a, b) / (refNorm(a) * refNorm(b));

double refDistance(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = a[i] - b[i];
    sum += d * d;
  }
  return sqrt(sum);
}

/// Tolerance for comparing a single-precision SIMD accumulation against
/// the double-precision reference: 1e-5 relative to the norm product,
/// which bounds the sum of absolute products (Cauchy-Schwarz).
double dotTolerance(Float32List a, Float32List b) =>
    1e-5 * (1 + refNorm(a) * refNorm(b));

/// Dimensions that exercise the SIMD lanes and every scalar tail length,
/// including sizes around the unroll width (16 components).
const tailDims = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 768, 1000, 1023, 1024];
