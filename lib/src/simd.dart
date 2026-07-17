// Internal SIMD kernels shared by the public operations and VectorMatrix.
// This file is not exported.

import 'dart:typed_data';

/// Returns [v] itself when its data starts at a 16-byte boundary of its
/// buffer, otherwise an aligned copy.
///
/// `Float32x4List.view` requires the byte offset to be a multiple of 16.
/// Freshly allocated lists always qualify; only sublist views into the
/// middle of another list can fail the check.
Float32List alignedView(Float32List v) =>
    (v.offsetInBytes & 15) == 0 ? v : Float32List.fromList(v);

/// Views the complete groups of four floats in [v] as SIMD lanes.
///
/// [v] must be 16-byte aligned (see [alignedView]). Components beyond
/// `v.length & ~3` are not covered and must be handled by scalar tails.
Float32x4List lanesOf(Float32List v) =>
    Float32x4List.view(v.buffer, v.offsetInBytes, v.length >> 2);

/// Index of the first NaN or infinite component of [v], or -1 when every
/// component is finite.
///
/// The scan is SIMD: `x - x` is zero exactly for finite `x` and NaN for
/// NaN or infinite `x`, so a lane-wise comparison against zero detects
/// non-finite components four at a time. Only when the combined mask
/// fails does a scalar pass locate the exact index.
int firstNonFinite(Float32List v) {
  final lanes = lanesOf(v);
  final zero = Float32x4.zero();
  var ok = Int32x4(-1, -1, -1, -1);
  for (var i = 0; i < lanes.length; i++) {
    final x = lanes[i];
    ok &= (x - x).equal(zero);
  }
  if (ok.signMask == 0xF) {
    for (var i = lanes.length << 2; i < v.length; i++) {
      if (!v[i].isFinite) return i;
    }
    return -1;
  }
  for (var i = 0; i < v.length; i++) {
    if (!v[i].isFinite) return i;
  }
  return -1;
}

/// Dot product over [count] SIMD lanes starting at [aStart] in [a] and
/// [bStart] in [b].
///
/// Uses four independent accumulators so consecutive fused
/// multiply-adds do not wait on each other's results (latency hiding).
/// Accumulation is single precision; see the package documentation for
/// the precision consequences.
double dotLanes(
  Float32x4List a,
  int aStart,
  Float32x4List b,
  int bStart,
  int count,
) {
  var acc0 = Float32x4.zero();
  var acc1 = Float32x4.zero();
  var acc2 = Float32x4.zero();
  var acc3 = Float32x4.zero();
  final unrolled = count & ~3;
  var i = 0;
  for (; i < unrolled; i += 4) {
    acc0 += a[aStart + i] * b[bStart + i];
    acc1 += a[aStart + i + 1] * b[bStart + i + 1];
    acc2 += a[aStart + i + 2] * b[bStart + i + 2];
    acc3 += a[aStart + i + 3] * b[bStart + i + 3];
  }
  for (; i < count; i++) {
    acc0 += a[aStart + i] * b[bStart + i];
  }
  final acc = (acc0 + acc1) + (acc2 + acc3);
  return acc.x + acc.y + acc.z + acc.w;
}

/// Squared Euclidean distance over [count] SIMD lanes, with the same
/// layout and accumulator scheme as [dotLanes].
double squaredDistanceLanes(
  Float32x4List a,
  int aStart,
  Float32x4List b,
  int bStart,
  int count,
) {
  var acc0 = Float32x4.zero();
  var acc1 = Float32x4.zero();
  var acc2 = Float32x4.zero();
  var acc3 = Float32x4.zero();
  final unrolled = count & ~3;
  var i = 0;
  for (; i < unrolled; i += 4) {
    final d0 = a[aStart + i] - b[bStart + i];
    final d1 = a[aStart + i + 1] - b[bStart + i + 1];
    final d2 = a[aStart + i + 2] - b[bStart + i + 2];
    final d3 = a[aStart + i + 3] - b[bStart + i + 3];
    acc0 += d0 * d0;
    acc1 += d1 * d1;
    acc2 += d2 * d2;
    acc3 += d3 * d3;
  }
  for (; i < count; i++) {
    final d = a[aStart + i] - b[bStart + i];
    acc0 += d * d;
  }
  final acc = (acc0 + acc1) + (acc2 + acc3);
  return acc.x + acc.y + acc.z + acc.w;
}

/// Dot product of two equal-length vectors: SIMD lanes plus a scalar
/// tail for the last `length % 4` components.
///
/// Both inputs must be 16-byte aligned (see [alignedView]).
double dotFull(Float32List a, Float32List b) {
  final laneCount = a.length >> 2;
  var sum = dotLanes(lanesOf(a), 0, lanesOf(b), 0, laneCount);
  for (var i = laneCount << 2; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// Squared Euclidean distance of two equal-length vectors: SIMD lanes
/// plus a scalar tail.
///
/// Both inputs must be 16-byte aligned (see [alignedView]).
double squaredDistanceFull(Float32List a, Float32List b) {
  final laneCount = a.length >> 2;
  var sum = squaredDistanceLanes(lanesOf(a), 0, lanesOf(b), 0, laneCount);
  for (var i = laneCount << 2; i < a.length; i++) {
    final d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/// Multiplies every component of [v] by [factor], in place.
///
/// Works on unaligned views too; those take the scalar path.
void scaleInPlace(Float32List v, double factor) {
  if ((v.offsetInBytes & 15) == 0) {
    final lanes = lanesOf(v);
    final f = Float32x4.splat(factor);
    for (var i = 0; i < lanes.length; i++) {
      lanes[i] *= f;
    }
    for (var i = lanes.length << 2; i < v.length; i++) {
      v[i] *= factor;
    }
  } else {
    for (var i = 0; i < v.length; i++) {
      v[i] *= factor;
    }
  }
}
