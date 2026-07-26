# What this package costs on the web

`vector_kit` is fast on the Dart VM because its inner loops use `Float32x4`,
which compiles to real SIMD instructions there. Off the native VM that type is
emulated, and the emulation is slower than not using SIMD at all.

The package declares `platform:web`, so here is the size of that, measured
rather than estimated. Reproduce it yourself:

```
dart test test/platform_cost_test.dart -t bench
dart test test/platform_cost_test.dart -t bench -p chrome
```

Apple M-series, Dart 3.11, 1000 rows × 384 dimensions, 200 queries after warmup:

| | native VM | Chrome (dart2js) |
|---|---:|---:|
| `topKCosine`, 1000×384 | **79 µs** | **4,794 µs** |

That is a factor of 60 on the operation the package exists for.

## Why

The second benchmark in that file isolates it: the same dot product written two
ways, over the same data.

| | SIMD (`Float32x4`) | plain loop (`Float32List`) | ratio |
|---|---:|---:|---:|
| native VM | 117 µs | 260 µs | SIMD **2.2× faster** |
| Chrome | 3,490 µs | 315 µs | SIMD **11× slower** |

Read the scalar column across rows: 260 µs native, 315 µs on Chrome. A plain
loop is close to platform-neutral. The whole gap is the emulated SIMD, which is
the one thing the package leans on.

That second table is a kernel microbenchmark, though, and a kernel is not a
search. It dot-products one cache-resident pair of vectors over and over: no
streaming memory traffic, no per-row divide, no top-k to maintain. So the third
benchmark measures the thing you would actually write instead of taking this
dependency, over the same 1000×384 corpus as the first one: one packed
`Float32List`, cached row norms, a k-sized insertion top-k, no SIMD anywhere.

| | native VM | Chrome (dart2js) | ratio |
|---|---:|---:|---:|
| `topKCosine` (this package) | 79 µs | 4,794 µs | **61×** |
| hand-written scalar scan | **257 µs** | **260 µs** | **1.01×** |

Minimum of six native runs and four Chrome runs; the spread inside each was
under 2%. Both platforms print the same top hit, so the two columns did the
same work rather than merely taking the same time.

The hand-written scan is platform-neutral to within one percent. That is the
cleanest statement of the problem: on the native VM this package is 3.3x faster
than the loop it replaces, and on Chrome it is 18x slower than that same loop.

## What this means for you

- **Server or mobile (Dart VM, AOT):** use it as documented. SIMD is doing what
  it is there for.
- **Flutter web or any dart2js/wasm target:** at a few thousand rows a plain
  loop over `Float32List` will beat this package today. Do not add the
  dependency for speed on the web; add it for the API or the int8 quantization
  if those help you, and measure your own case.

## Fixing it

The fix is a conditional kernel: keep the SIMD path on native, dispatch to a
scalar path off it, chosen with `if (dart.library.js_interop)` at import time.
The measurement above says the scalar path lands near 315 µs on Chrome, so the
expected result is roughly an 11× web improvement with native untouched.

The storage layout does not have to change — rows already live in one
`Float32List` and `Float32x4List` is only a view over it — so this is contained
to `lib/src/simd.dart` and its call sites.

Tracked, not done. This document exists because the gap was shipped for weeks
behind a `platform:web` badge that nothing tested.

CI now runs the Chrome suite and a wasm compile on every push, which means the
web target can no longer break without anyone noticing. Be clear about what that
does not cover: the benchmark above asserts nothing about time, because CI
runners vary too much for any threshold to be meaningful. It prints. A
regression from 4,794 µs to 50,000 µs would leave CI green. The guard is against
breakage; the numbers are a measurement you have to take.
