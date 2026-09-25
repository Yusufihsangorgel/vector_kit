# Package engineering rules: vector_kit

Rules-Version: vector_kit/76e09d4c66fb981e4e3f813854a2958d708b13943a08397f04c70209eaca6676
Core-Version: 1
Core-Digest: 1825fa7ff346dca23e65b1b3bf9b2e3e06959f1414bae9952d596d2f62f09b8f
Survey-Digest: f90f45c8a172068c3ed3b9488ba5a7cb4e58efa93c380d2d9a70b399349ec35e
Evidence-Revision: 50e9f3f
Verified-Revision: unverified

Read CONTRIBUTING.md and docs/engineering/debt.json before editing.

## Current architecture
A numeric core with zero runtime dependencies. A single barrel (lib/vector_kit.dart) exposes two implementation files: ops.dart (pairwise vector functions) and vector_matrix.dart (VectorMatrix: padded rows, cached norm, bounded-heap top-k; QuantizedMatrix: int8 memory savings). Platform isolation uses conditional exports: simd.dart picks simd_native.dart (Float32x4) on the VM and simd_web.dart (scalar) on the web. The two files carry the same names and signatures. Performance claims are measured with bench/, benchmark/, and tool/ scripts and linked under doc/. The rag_kit VectorStore adapter lives in example/ on purpose. rag_kit is a dev dependency. Scanned HEAD: 50e9f3f (version 1.4.0).

## Layers and responsibilities
- lib/vector_kit.dart: Five free functions and two matrix types with `show`. The library dartdoc explains the precision and validation contract.
- lib/src/ops.dart, lib/src/vector_matrix.dart: Validation, kernel calls, top-k heap, VKT1 binary format, int8 quantization.
- lib/src/simd.dart: Conditional export by compile target. Not exported.
- lib/src/simd_native.dart, lib/src/simd_web.dart: Float32x4 kernels with four accumulators on the VM, scalar kernels accumulating doubles on the web. Same names and signatures.
- bench/, benchmark/, tool/: README numbers, break-even point scan, quantization cost, platform charts. tool/euclid_probe.dart is a temporary file (debt).
- example/, test/ (helpers.dart): Semantic search examples, the rag_kit adapter and its contract tests, cross-platform ordering tests.

## Public API and dependency direction
lib/vector_kit.dart exposes with `show`: cosineSimilarity, dot, euclideanDistance, normalized, normalizeInPlace (ops.dart) and QuantizedMatrix, VectorMatrix (vector_matrix.dart) (vector_kit.dart:22-24). simd*.dart is not exported (simd.dart:1-2). There is no extension point: two final classes and free functions. The wire format is VKT1 (vector_matrix.dart:322-326). Pairwise vector functions take only Float32List. Top-k queries take List<double>. rowAt returns a live view (documented, 202-215). QuantizedMatrix has no add/rowAt/toBytes (AGENTS.md contract).

ops.dart → dart:math, dart:typed_data, simd.dart (1-4). vector_matrix.dart → dart:math, dart:typed_data, simd.dart (1-4). simd.dart → conditionally simd_native.dart | simd_web.dart (13-15). Kernel files import only dart:typed_data. Nothing imports simd_native.dart or simd_web.dart directly. No runtime dependencies. rag_kit is dev only (example/vector_kit_store.dart and test/rag_kit_store_test.dart).

## Error, state and platform contracts
- Platform isolation with conditional exports. Same names and `typedef Lanes` in both files (simd.dart:1-15; simd_native.dart:1-15; simd_web.dart:1-23).
- Validation through accumulation: a finite squared norm proves the input is finite. A rescan happens only on failure (ops.dart:49-56, 91-95; vector_matrix.dart:178-197).
- Eager validation: ArgumentError.value with the parameter name, FormatException on corrupt bytes (vector_kit.dart:16-19; ops.dart:6-17; vector_matrix.dart:67-139).
- Packed padded row layout, cached norm, bounded min-heap top-k (vector_matrix.dart:6-18, 390-461).
- Validation of the actual payload size before allocation (vector_matrix.dart:89-109).
- The sibling package adapter lives in example/. Contract tests import the example (test/rag_kit_store_test.dart; AGENTS.md:134-138).
- Measured claim: comments cite the number and doc/web-performance.md (simd.dart:4-8; simd_web.dart:8-9).
- Public types are final (vector_matrix.dart:19, 478; commit 71151af).

## Package rules
### vector_kit/VEC-01 [MUST]
Only lib/src/simd.dart imports simd_native.dart or simd_web.dart. Everything else imports simd.dart.
Reason: Call sites must not know which core they get. Platform selection stays in one place.
Evidence: lib/src/simd.dart:1-15; lib/src/simd_native.dart:1-2; lib/src/simd_web.dart:1-2; lib/src/ops.dart:4; lib/src/vector_matrix.dart:4; AGENTS.md:149
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-02 [MUST]
A kernel added or changed in simd_native.dart or simd_web.dart gets the same name and signature in the other file in the same commit, and the VM, dart2js and dart2wasm CI runs pass.
Reason: A conditional export breaks on the platform that is not tested at that moment.
Evidence: lib/src/simd_native.dart:4-6; lib/src/simd_web.dart:11-12; .github/workflows/ci.yaml:30-43; AGENTS.md:130-131
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-03 [MUST_NOT]
Do not use `Float32x4` or any other SIMD type on the web path.
Reason: On the web, emulation was measured 15-41x slower than the scalar loop. 1.1.0 was split for this reason.
Evidence: lib/src/simd_web.dart:4-9; lib/src/simd.dart:4-10; AGENTS.md:128-129; commit 0a8d162
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-04 [MUST]
Public entry points validate eagerly: empty input, a length mismatch, a non-finite component, and a zero vector where the result is undefined throw `ArgumentError` naming the parameter. No score is NaN.
Reason: The library contract guarantees that NaN never spreads into the scores.
Evidence: lib/vector_kit.dart:16-19; lib/src/ops.dart:6-17, 58-59, 84-86; lib/src/vector_matrix.dart:160-200, 232-253
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-05 [MUST]
Top-k rankings stay identical across VM and web. test/cross_platform_test.dart pins exact rows and runs on all three CI targets.
Reason: Scores may differ by about 5e-9; the ordering cannot. As the cores drift apart, only this test catches it.
Evidence: test/cross_platform_test.dart:55-68; .github/workflows/ci.yaml:30-40; lib/vector_kit.dart:11-14
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-06 [MUST_NOT]
Do not add a runtime dependency. The rag_kit adapter stays in example/ with rag_kit as a dev dependency.
Reason: If the adapter goes into lib, it becomes a runtime dependency of every rag_kit consumer.
Evidence: pubspec.yaml:23-32; AGENTS.md:134-138
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-07 [MUST]
A change to the VKT1 format validates the declared size against the real payload before allocating, and `toBytes` never writes row padding.
Reason: Hostile uint32 values can cause very large allocations. Padding is an internal detail.
Evidence: lib/src/vector_matrix.dart:67-109, 322-342
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-08 [SHOULD]
A performance claim in dartdoc, a comment or the README names the script under bench/, benchmark/ or tool/ that measured it.
Reason: All speed claims of the package depend on measurement. A claim without measurement is technical debt.
Evidence: lib/src/simd.dart:4-8; lib/src/simd_web.dart:8-9; bench/bench.dart:1-10; benchmark/quantization_benchmark.dart:1-8
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-09 [MUST_NOT]
Do not commit temporary probes, reproductions or snapshots of unfinished work. A reproduction becomes a named regression test or is deleted before the commit.
Reason: Two TEMPORARY files entered public history through a snapshot commit. One copies lib logic.
Evidence: test/euclidean_repro_test.dart:1; tool/euclid_probe.dart:1-8; commit 34e84e4
Evidence role: counterexample
Existing violation: vector_kit-D001, vector_kit-D002

### vector_kit/VEC-10 [MUST]
Public types are `final class`. A method that returns a live view says so in its dartdoc.
Reason: The API was frozen in 1.0 and the classes were sealed. A live view can leave the cached norm stale.
Evidence: lib/src/vector_matrix.dart:19, 202-215, 478; commit 71151af
Evidence role: current-pattern
Existing violation: none

### vector_kit/VEC-11 [MUST]
A behavior change lands with a CHANGELOG entry and a test in the same commit.
Reason: Repository practice. The snapshot commit is the only violation of this rule.
Evidence: commit 7dd57bd, 6d5d0ac, ca6468d, 0a8d162 (CHANGELOG + test); counterexample 34e84e4
Evidence role: both
Existing violation: vector_kit-D001

### vector_kit/VEC-12 [MUST]
Export public names with a `show` list from lib/vector_kit.dart. Tests import that library, their helpers and example files only.
Reason: The public API boundary is visible in a single file.
Evidence: lib/vector_kit.dart:22-24; test imports (package:vector_kit/vector_kit.dart, helpers.dart, ../example/vector_kit_store.dart)
Evidence role: current-pattern
Existing violation: none

## Required verification
- Working directory: repository root; command: dart pub get; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:22.
- Working directory: repository root; command: dart format --output=none --set-exit-if-changed .; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:24.
- Working directory: repository root; command: dart analyze --fatal-infos; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:26.
- Working directory: repository root; command: dart test; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:28.
- Working directory: repository root; command: dart test -p chrome; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:37.
- Working directory: repository root; command: dart test -p chrome -c dart2wasm; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:40.
- Working directory: repository root; command: dart compile wasm example/semantic_search.dart -o /tmp/vk.wasm; conditions: ci.yaml job build; evidence: .github/workflows/ci.yaml:43.
Not verified by the survey:
- The scan used the local HEAD (50e9f3f). Equality with origin and uncommitted changes were not measured.
- That the simd.dart:15 branch is truly unreachable was not verified by compiling.
- Whether the euclidean_repro_test claim is covered one-to-one in quantized_matrix_test was not compared (only 14 topKEuclidean uses were counted).
- Cognitive complexity scores were not measured. Candidate: vector_matrix.dart:72-139 (fromBytes).
- Whether tests pass today on all three targets: `dart test` was not run.

## Existing debt
The complete register is docs/engineering/debt.json.
- vector_kit-D001 | small | test/euclidean_repro_test.dart:1-19 | temporary file committed
  Fix: Delete the file. If its claim is absent from quantized_matrix_test.dart, move it there as a named regression test.
  Closure: test/euclidean_repro_test.dart is deleted. Any claim it makes that quantized_matrix_test.dart lacks exists there as a named regression test.
- vector_kit-D002 | small | tool/euclid_probe.dart:1-8 | temporary file + duplicated logic
  Fix: Delete the file.
  Closure: tool/euclid_probe.dart is deleted.
- vector_kit-D003 | small | lib/src/vector_matrix.dart:536-537 | stale dartdoc
  Fix: Fix it to `matrix.rowCount * matrix.dimension * 4`.
  Closure: The dartdoc at vector_matrix.dart:536-537 names matrix.rowCount instead of matrix.length.
- vector_kit-D004 | small | lib/src/simd.dart:15 | probable dead code
  Fix: Delete the line. The item closes if the chrome dart2js and dart2wasm jobs stay green.
  Closure: The dart.library.js branch is gone from simd.dart and the chrome dart2js and dart2wasm CI jobs stay green.
- vector_kit-D005 | small | bench/ (bench.dart, break_even.dart, naive.dart) and benchmark/ (quantization_benchmark.dart) | scattered layout
  Fix: Gather them in one directory and update the README and AGENTS.md references.
  Closure: All benchmark scripts live in one directory and the README and AGENTS.md references point to it.
- vector_kit-D006 | small | test/platform_cost_test.dart:1 | undefined test tag
  Fix: Define the `bench` tag in dart_test.yaml. If CI runs it on purpose, state this in ci.yaml; if it should not run, add `-x bench`.
  Closure: dart_test.yaml defines the bench tag and package:test prints no warning. ci.yaml either states that CI runs the tag on purpose or adds -x bench.
- vector_kit-D007 | small | test/cross_platform_test.dart:66-67, 85-86 | lint suppression
  Fix: Use `printOnFailure` (no ignore needed).
  Closure: cross_platform_test.dart prints diagnostics with printOnFailure and contains no ignore: avoid_print comment.
- vector_kit-D008 | small | lib/src/vector_matrix.dart:390-660 | file size / single responsibility
  Fix: Move QuantizedMatrix to its own lib/src file and the heap to a shared private file. The exports stay the same.
  Closure: QuantizedMatrix lives in its own lib/src file and _TopKHeap in a shared private file with the exports unchanged. The package tests pass.
- vector_kit-D009 | small | lib/src/ops.dart:66-73, 141-148 | duplicated logic
  Fix: A single helper for pairs. Safety net: ops_test.dart.
  Closure: One shared helper checks non-finite components for vector pairs and both ops.dart sites call it. ops_test.dart passes.
