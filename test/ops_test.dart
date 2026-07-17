import 'dart:math';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:vector_kit/vector_kit.dart';

import 'helpers.dart';

void main() {
  group('dot', () {
    test('matches a hand-computed value', () {
      final a = Float32List.fromList([1, 2, 3]);
      final b = Float32List.fromList([4, 5, 6]);
      expect(dot(a, b), 32.0);
      expect(dot(a, a), 14.0);
    });

    test('handles every tail length', () {
      final rng = Random(1);
      for (final dim in tailDims) {
        final a = randomVector(dim, rng);
        final b = randomVector(dim, rng);
        expect(
          dot(a, b),
          closeTo(refDot(a, b), dotTolerance(a, b)),
          reason: 'dim $dim',
        );
      }
    });

    test('matches the double reference on 100 random vectors', () {
      final rng = Random(2);
      for (var i = 0; i < 100; i++) {
        final dim = 1 + rng.nextInt(1024);
        final a = randomVector(dim, rng);
        final b = randomVector(dim, rng);
        expect(
          dot(a, b),
          closeTo(refDot(a, b), dotTolerance(a, b)),
          reason: 'iteration $i, dim $dim',
        );
      }
    });

    test('rejects mismatched lengths', () {
      final a = Float32List(4);
      final b = Float32List(5);
      expect(() => dot(a, b), throwsArgumentError);
    });

    test('rejects empty vectors', () {
      expect(() => dot(Float32List(0), Float32List(0)), throwsArgumentError);
    });

    test('rejects NaN components in either argument', () {
      final ok = Float32List.fromList([1, 2, 3, 4, 5]);
      final bad = Float32List.fromList([1, 2, double.nan, 4, 5]);
      expect(() => dot(bad, ok), throwsArgumentError);
      expect(() => dot(ok, bad), throwsArgumentError);
    });

    test('rejects infinite components, including in the scalar tail', () {
      final ok = Float32List.fromList([1, 2, 3, 4, 5]);
      final infLane = Float32List.fromList([double.infinity, 2, 3, 4, 5]);
      final infTail = Float32List.fromList([
        1,
        2,
        3,
        4,
        double.negativeInfinity,
      ]);
      expect(() => dot(infLane, ok), throwsArgumentError);
      expect(() => dot(ok, infTail), throwsArgumentError);
    });

    test('returns infinity when finite inputs overflow the accumulator', () {
      // 3e38 is finite in float32; its square is not. With all inputs
      // finite this is accumulator overflow, not invalid input.
      final huge = Float32List.fromList([3e38, 3e38, 3e38, 3e38]);
      expect(dot(huge, huge), double.infinity);
    });

    test('accepts unaligned sublist views', () {
      final rng = Random(3);
      final backing = randomVector(70, rng);
      // Offset of one float (4 bytes) breaks 16-byte alignment.
      final a = Float32List.sublistView(backing, 1, 34);
      final b = Float32List.sublistView(backing, 34, 67);
      final aCopy = Float32List.fromList(a);
      final bCopy = Float32List.fromList(b);
      expect(dot(a, b), dot(aCopy, bCopy));
    });
  });

  group('cosineSimilarity', () {
    test('matches a hand-computed value', () {
      final a = Float32List.fromList([1, 0]);
      final b = Float32List.fromList([1, 1]);
      expect(cosineSimilarity(a, b), closeTo(1 / sqrt2, 1e-6));
    });

    test('is 1 for a vector against itself and against a scaled copy', () {
      final rng = Random(4);
      final v = randomVector(129, rng);
      final scaled = Float32List.fromList([for (final x in v) x * 2]);
      expect(cosineSimilarity(v, v), closeTo(1, 1e-6));
      expect(cosineSimilarity(v, scaled), closeTo(1, 1e-6));
      expect(cosineSimilarity(v, v), lessThanOrEqualTo(1));
      expect(cosineSimilarity(v, scaled), lessThanOrEqualTo(1));
    });

    test('is -1 for opposite vectors and 0 for orthogonal ones', () {
      final a = Float32List.fromList([2, -1, 3]);
      final opposite = Float32List.fromList([-2, 1, -3]);
      expect(cosineSimilarity(a, opposite), closeTo(-1, 1e-6));
      expect(cosineSimilarity(a, opposite), greaterThanOrEqualTo(-1));

      final x = Float32List.fromList([1, 0, 0, 0, 0]);
      final y = Float32List.fromList([0, 0, 0, 0, 1]);
      expect(cosineSimilarity(x, y), closeTo(0, 1e-9));
    });

    test('matches the double reference on 100 random vectors', () {
      final rng = Random(5);
      for (var i = 0; i < 100; i++) {
        final dim = 1 + rng.nextInt(1024);
        final a = randomVector(dim, rng);
        final b = randomVector(dim, rng);
        expect(
          cosineSimilarity(a, b),
          closeTo(refCosine(a, b), 1e-5),
          reason: 'iteration $i, dim $dim',
        );
      }
    });

    test('rejects zero vectors on either side', () {
      final v = Float32List.fromList([1, 2, 3]);
      final zero = Float32List(3);
      expect(() => cosineSimilarity(zero, v), throwsArgumentError);
      expect(() => cosineSimilarity(v, zero), throwsArgumentError);
    });

    test('rejects mismatched lengths and non-finite components', () {
      final v = Float32List.fromList([1, 2, 3]);
      expect(() => cosineSimilarity(v, Float32List(4)), throwsArgumentError);
      final nan = Float32List.fromList([1, double.nan, 3]);
      expect(() => cosineSimilarity(v, nan), throwsArgumentError);
    });

    test('rejects finite components whose squared norm overflows', () {
      // 3e38 is finite in float32, but its square is not.
      final huge = Float32List.fromList([3e38, 0, 0, 0]);
      final v = Float32List.fromList([1, 0, 0, 0]);
      expect(() => cosineSimilarity(huge, v), throwsArgumentError);
    });
  });

  group('euclideanDistance', () {
    test('matches a hand-computed value', () {
      final a = Float32List.fromList([0, 0]);
      final b = Float32List.fromList([3, 4]);
      expect(euclideanDistance(a, b), 5.0);
    });

    test('is 0 from a vector to itself and symmetric', () {
      final rng = Random(6);
      final a = randomVector(100, rng);
      final b = randomVector(100, rng);
      expect(euclideanDistance(a, a), 0.0);
      expect(euclideanDistance(a, b), euclideanDistance(b, a));
    });

    test('handles every tail length', () {
      final rng = Random(7);
      for (final dim in tailDims) {
        final a = randomVector(dim, rng);
        final b = randomVector(dim, rng);
        final ref = refDistance(a, b);
        expect(
          euclideanDistance(a, b),
          closeTo(ref, 1e-5 * (1 + ref)),
          reason: 'dim $dim',
        );
      }
    });

    test('matches the double reference on 100 random vectors', () {
      final rng = Random(8);
      for (var i = 0; i < 100; i++) {
        final dim = 1 + rng.nextInt(1024);
        final a = randomVector(dim, rng);
        final b = randomVector(dim, rng);
        final ref = refDistance(a, b);
        expect(
          euclideanDistance(a, b),
          closeTo(ref, 1e-5 * (1 + ref)),
          reason: 'iteration $i, dim $dim',
        );
      }
    });

    test('returns infinity when finite inputs overflow the accumulator', () {
      final a = Float32List.fromList([3e38, 0, 0, 0]);
      final b = Float32List.fromList([-3e38, 0, 0, 0]);
      expect(euclideanDistance(a, b), double.infinity);
    });

    test('rejects mismatched lengths, empty input, and NaN', () {
      final v = Float32List.fromList([1, 2, 3]);
      expect(() => euclideanDistance(v, Float32List(2)), throwsArgumentError);
      expect(
        () => euclideanDistance(Float32List(0), Float32List(0)),
        throwsArgumentError,
      );
      final nan = Float32List.fromList([double.nan, 2, 3]);
      expect(() => euclideanDistance(nan, v), throwsArgumentError);
    });
  });

  group('normalizeInPlace', () {
    test('matches a hand-computed value', () {
      final v = Float32List.fromList([3, 4]);
      normalizeInPlace(v);
      expect(v[0], closeTo(0.6, 1e-7));
      expect(v[1], closeTo(0.8, 1e-7));
    });

    test('produces unit norm and preserves direction at every tail '
        'length', () {
      final rng = Random(9);
      for (final dim in tailDims) {
        final v = randomVector(dim, rng, scale: 10);
        final before = Float32List.fromList(v);
        normalizeInPlace(v);
        expect(refNorm(v), closeTo(1, 1e-5), reason: 'dim $dim');
        final norm = refNorm(before);
        for (var i = 0; i < dim; i++) {
          expect(
            v[i],
            closeTo(before[i] / norm, 1e-6),
            reason: 'dim $dim, component $i',
          );
        }
      }
    });

    test('rejects the zero vector, empty input, and NaN', () {
      expect(() => normalizeInPlace(Float32List(8)), throwsArgumentError);
      expect(() => normalizeInPlace(Float32List(0)), throwsArgumentError);
      expect(
        () => normalizeInPlace(Float32List.fromList([1, double.nan])),
        throwsArgumentError,
      );
    });

    test('rejects finite components whose squared norm overflows, '
        'leaving the input unchanged', () {
      final huge = Float32List.fromList([3e38, 1, 2, 3]);
      expect(() => normalizeInPlace(huge), throwsArgumentError);
      expect(huge[1], 1.0);
    });

    test('normalizes unaligned sublist views in place', () {
      final backing = Float32List.fromList([9, 3, 4, 0, 0]);
      final view = Float32List.sublistView(backing, 1, 3);
      normalizeInPlace(view);
      expect(view[0], closeTo(0.6, 1e-7));
      expect(view[1], closeTo(0.8, 1e-7));
      expect(backing[0], 9.0);
      expect(backing[3], 0.0);
    });
  });

  group('normalized', () {
    test('returns a unit-norm copy and leaves the input untouched', () {
      final v = Float32List.fromList([3, 4]);
      final unit = normalized(v);
      expect(unit[0], closeTo(0.6, 1e-7));
      expect(unit[1], closeTo(0.8, 1e-7));
      expect(v[0], 3.0);
      expect(v[1], 4.0);
      expect(identical(unit, v), isFalse);
    });

    test('rejects the zero vector', () {
      expect(() => normalized(Float32List(3)), throwsArgumentError);
    });
  });
}
