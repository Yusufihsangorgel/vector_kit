// Scalar kernels for every target that is not the Dart VM. Reached through
// `simd.dart`; never imported directly.
//
// `Float32x4` exists off the VM but is emulated: dart2js backs it with four
// boxed doubles and allocates a fresh object per lane read and per arithmetic
// operation, and dart2wasm's own patch file calls its version `NaiveFloat32x4`.
// The emulation is slower than doing no SIMD at all, so this file does no SIMD.
// Measured, 1000x384: `topKCosine` 4,794 us through the emulated kernels
// against 260 us for the scalar scan below. See `doc/web-performance.md`.
//
// Every name and signature here mirrors `simd_native.dart`. The one deliberate
// behavioural difference is accumulator width, documented on [dotLanes].

import 'dart:typed_data';

/// A view over a [Float32List] in the form the kernels want.
///
/// The list itself here; four-float SIMD lanes on the VM. Lane indices and
/// counts passed to [dotLanes] and [squaredDistanceLanes] are in units of this
/// view, so callers do not have to know which one they are holding. A lane is
/// four components either way, which is why the scalar kernels multiply the
/// indices they are given by four.
typedef Lanes = Float32List;

/// Smallest magnitude that rounds to infinity in float32, which is 2^128.
///
/// Values strictly below this round down to `float32.maxFinite`; values at or
/// above it become infinite. Used to keep the overflow contract identical to
/// the VM (see [_asFloat32Sum]).
const double _overflowsFloat32 = 3.4028236692093846e38;

/// Reports a double accumulation the way a float32 accumulator would.
///
/// The VM kernels accumulate in single precision, so a sum too large for
/// float32 arrives at the caller as infinity and is reported as an overflow.
/// A double accumulator would carry that same sum finitely and let it pass
/// validation, so the contract, not just the number, would differ by platform.
/// Collapsing the out-of-range case back to infinity keeps
/// `VectorMatrix.add`'s "squared norm overflows single precision" behaviour
/// the same on both.
///
/// This catches overflow in the total. A signed dot product whose partial sums
/// leave float32 range but whose total does not can still differ: the VM
/// saturates to infinity mid-loop and stays there, while this returns the
/// finite double answer. That case needs components near 1e38 and is left
/// alone deliberately, since the finite answer is the correct one.
double _asFloat32Sum(double sum) {
  if (sum.abs() < _overflowsFloat32) return sum;
  if (sum.isNaN) return sum;
  return sum.isNegative ? double.negativeInfinity : double.infinity;
}

/// Returns [v] unchanged.
///
/// The VM needs a 16-byte aligned copy before it can build a `Float32x4List`
/// view. Nothing here reads through such a view, so the copy is pure cost and
/// is skipped.
Float32List alignedView(Float32List v) => v;

/// Returns [v] unchanged; there is no separate lane view to build.
Lanes lanesOf(Float32List v) => v;

/// Index of the first NaN or infinite component of [v], or -1 when every
/// component is finite.
int firstNonFinite(Float32List v) {
  for (var i = 0; i < v.length; i++) {
    if (!v[i].isFinite) return i;
  }
  return -1;
}

/// Dot product over [count] lanes starting at lane [aStart] in [a] and lane
/// [bStart] in [b], where a lane is four components.
///
/// Accumulates in double, unlike the VM kernels, which accumulate in float32
/// because that is what their vector registers hold. Reading a [Float32List]
/// element already widens it to a double, so keeping the running sum narrow
/// would mean rounding through memory on every step and would cost more than
/// the SIMD emulation this file exists to avoid. The results are therefore
/// close but not bit-identical across platforms, with the web being the more
/// accurate of the two. `doc/web-performance.md` carries the measured size of
/// that difference and its effect on ranking.
double dotLanes(Lanes a, int aStart, Lanes b, int bStart, int count) {
  var i = aStart << 2;
  var j = bStart << 2;
  final end = i + (count << 2);
  var sum = 0.0;
  while (i < end) {
    sum += a[i] * b[j];
    i++;
    j++;
  }
  return _asFloat32Sum(sum);
}

/// Squared Euclidean distance over [count] lanes, with the same layout and
/// accumulator width as [dotLanes].
double squaredDistanceLanes(
  Lanes a,
  int aStart,
  Lanes b,
  int bStart,
  int count,
) {
  var i = aStart << 2;
  var j = bStart << 2;
  final end = i + (count << 2);
  var sum = 0.0;
  while (i < end) {
    final d = a[i] - b[j];
    sum += d * d;
    i++;
    j++;
  }
  return _asFloat32Sum(sum);
}

/// Dot product of two equal-length vectors.
double dotFull(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return _asFloat32Sum(sum);
}

/// Squared Euclidean distance of two equal-length vectors.
double squaredDistanceFull(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = a[i] - b[i];
    sum += d * d;
  }
  return _asFloat32Sum(sum);
}

/// Multiplies every component of [v] by [factor], in place.
void scaleInPlace(Float32List v, double factor) {
  for (var i = 0; i < v.length; i++) {
    v[i] *= factor;
  }
}
