# What this package costs on the web

Short version: as of 1.1.0 it costs about what a hand-written loop costs. Before
1.1.0 it cost 15 to 41 times more, and this document exists because that shipped
for weeks behind a `platform:web` badge that nothing tested.

## The problem

`vector_kit`'s inner loops were written around `Float32x4`, which is a real SIMD
type only on the Dart VM. Off the VM the SDK emulates it: dart2js backs it with
four boxed doubles and allocates a fresh object on every lane read and every
arithmetic operation, and dart2wasm's own patch file names its version
`NaiveFloat32x4` with a `TODO` to implement real Wasm SIMD later.

Emulated SIMD is slower than doing no SIMD at all. So 1.1.0 keeps the SIMD
kernels for the VM and dispatches to plain scalar kernels everywhere else, using
a conditional import on `dart.library.js_interop`. Nothing in the public API
changed.

## Measured

Apple M-series, Dart 3.11, 1000 rows × 384 dimensions, `topKCosine`, k=10.
Minimum of several process runs per cell; the spread inside each was under 2%.
Reproduce it:

```
dart test test/platform_cost_test.dart -t bench
dart test test/platform_cost_test.dart -t bench -p chrome
dart test test/platform_cost_test.dart -t bench -p chrome -c dart2wasm
```

| | 1.0.4 | 1.1.0 | |
|---|---:|---:|---|
| native VM | 79 µs | **77 µs** | unchanged, still SIMD |
| Chrome, dart2js | 4,780 µs | **322 µs** | 14.8× faster |
| Chrome, dart2wasm | 12,102 µs | **292 µs** | 41.4× faster |

The third benchmark in that file is the honest yardstick: the same search
written by hand with no package at all, one packed `Float32List`, cached row
norms, a k-sized insertion top-k. It costs 257 µs native, 258 µs on dart2js and
272 µs on dart2wasm — a plain loop is close to platform-neutral. Against it:

| | 1.0.4 | 1.1.0 |
|---|---|---|
| native VM | 3.3× faster than the loop | **3.3× faster** |
| dart2js | 18× slower | **1.25× slower** |
| dart2wasm | 45× slower | **1.08× slower** |

So on the web the package no longer costs you anything meaningful against
writing it yourself, and you keep the API, the persistence and the int8 path.
It is still the VM where the SIMD earns its keep.

## The one behavioural difference

The VM kernels accumulate in float32, because that is what their vector
registers hold. The scalar kernels accumulate in double, because reading a
`Float32List` element already widens it and narrowing the running sum again
would mean rounding through memory on every step — which would cost more than
the emulation this change exists to avoid.

That makes web results slightly different from VM results, and slightly more
accurate. Measured on the corpus in `test/cross_platform_test.dart`: the top
cosine score is `0.1738678079298492` on the VM and `0.17386781263674544` on
both web backends, a difference of **4.7e-9**.

**Ranking is unaffected, and that is asserted rather than assumed.**
`test/cross_platform_test.dart` pins the exact top-10 row indices for cosine,
dot and euclidean, produced on the VM, and CI runs it on the VM, dart2js and
dart2wasm. All three return the same rows in the same order. If a future change
makes them diverge, the suite fails.

One consequence worth naming: a sum too large for float32 arrives at the caller
as infinity on the VM, and `VectorMatrix.add` reports it as an overflow. A
double accumulator would carry that same sum finitely and let the row through,
so the scalar kernels fold out-of-range totals back to infinity. Without that,
the *error contract* and not merely the number would differ by platform. There
is a test for it.

## What CI guards

Every push runs the full suite on the VM, on Chrome via dart2js, and on Chrome
via dart2wasm, plus a wasm compile. So the web target cannot break unnoticed and
the three platforms cannot silently start ranking differently.

CI does **not** assert timings: runners vary too much for a threshold to mean
anything, so the benchmark prints rather than fails. A regression from 322 µs
back to 4,800 µs would leave CI green. The guard is against breakage and
divergence; the numbers above are a measurement you have to take.
