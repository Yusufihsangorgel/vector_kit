// Selects the kernel implementation for the platform this is compiled for.
// Not exported.
//
// `Float32x4` is a real SIMD type only on the Dart VM. Everywhere else the
// SDK emulates it with four boxed doubles, and that emulation is slower than
// not using SIMD at all: measured on 1000x384, `topKCosine` costs 79 us
// natively and 4,794 us on Chrome, while the same search written as a plain
// scalar loop costs 257 us and 260 us. See `doc/web-performance.md`.
//
// So the SIMD kernels ship for the VM and a scalar set ships everywhere else.
// Both expose the same names and the same `Lanes` type alias, so no call site
// knows which one it got.
export 'simd_native.dart'
    if (dart.library.js_interop) 'simd_web.dart'
    if (dart.library.js) 'simd_web.dart';
