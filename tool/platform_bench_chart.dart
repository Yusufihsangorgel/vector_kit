// Generates doc/platform.svg and doc/platform.png, the platform chart in the
// README.
//
//   dart run tool/platform_bench_chart.dart
//
// The SVG is written unconditionally. The PNG is rasterized with
// `rsvg-convert` if it is on PATH (`brew install librsvg`); otherwise the
// command to run is printed and the SVG is left for you to convert.
//
// The numbers below are not measured here. They are transcribed from runs of
// `test/platform_cost_test.dart` and are the same figures
// `doc/web-performance.md` quotes, so the chart cannot drift away from the
// measurement by accident: changing the chart means re-running the benchmark
// and pasting new figures, which is the edit a reviewer would ask for anyway.
//
// The ratios the chart plots are computed from those microsecond figures
// rather than transcribed alongside them, so a mistyped ratio cannot disagree
// with the bar it labels.
import 'dart:io';
import 'dart:math' as math;

/// Microseconds per `topKCosine` query, k=10, 1000 rows of 384 dimensions.
///
/// Apple M-series, Dart 3.11, minimum of several process runs per cell with
/// under 2% spread inside each. Produced by:
///
///     dart test test/platform_cost_test.dart -t bench
///     dart test test/platform_cost_test.dart -t bench -p chrome
///     dart test test/platform_cost_test.dart -t bench -p chrome -c dart2wasm
///
/// [emulated] is what the package shipped through 1.0.4: `Float32x4` kernels
/// compiled for every target. [conditional] is 1.1.0 and later. [handWritten]
/// is the third benchmark in that file, the same search written out by a
/// caller with no package at all, and it is the denominator for every ratio on
/// the chart.
const _targets = [
  _Target(
    name: 'native VM',
    detail: 'JIT',
    emulated: 79,
    conditional: 77,
    handWritten: 257,
  ),
  _Target(
    name: 'Chrome',
    detail: 'dart2js',
    emulated: 4780,
    conditional: 322,
    handWritten: 258,
  ),
  _Target(
    name: 'Chrome',
    detail: 'dart2wasm',
    emulated: 12102,
    conditional: 292,
    handWritten: 272,
  ),
];

const _title = 'Float32x4 is a real SIMD type only on the Dart VM';
const _subtitleA =
    'One top-k search on three targets, each measured against the same search '
    'written by hand.';
const _subtitleB =
    'topKCosine, k=10, 1000 rows x 384 dims. The line is that hand-written '
    'loop at 1x.';

const _legendEmulated = 'Float32x4 kernels on every target';
const _legendConditional = 'SIMD on the VM, scalar kernels elsewhere';

const _footer = [
  'dart2js backs Float32x4 with four boxed doubles. dart2wasm names its own '
      'version NaiveFloat32x4.',
  'vector_kit 1.1.0 picks the kernel set from dart.library.js_interop at '
      'compile time.',
  'CI runs all three targets and pins the exact top-10 rows against '
      'divergence.',
];

// Canvas. Rendered at 2x so the PNG lands at 1520x1300, close enough to
// square for pub.dev's 190x190 search-card thumbnail, which fits rather than
// crops.
const _width = 760.0;
const _height = 650.0;
const _scale = 2;

// Plot area. The left gutter carries the target names.
const _plotLeft = 268.0;
const _plotRight = 636.0;
const _plotTop = 160.0;
const _axisY = 470.0;

// Log-scale domain, wide enough to hold 0.30x and 44.5x with margin.
const _domainMin = 0.22;
const _domainMax = 62.0;
const _ticks = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

// Group geometry.
const _groupPitch = 112.0;
const _barHeight = 26.0;
const _rowGap = 12.0;

// Palette, sampled from doc/bench.png so the two charts read as one set:
// Tailwind slate for the ground, one sky accent for what the package ships,
// one amber for the trap.
const _paper = '#0B1220';
const _ink = '#E2E8F0';
const _body = '#94A3B8';
const _muted = '#64748B';
const _grid = '#2B384B';
const _rule = '#475569';
const _sky = '#38BDF8';
const _amber = '#F59E0B';

const _mono = 'Menlo, monospace';

void main() {
  if (!Directory('doc').existsSync()) {
    stderr.writeln(
      'run this from the package root: doc/ not found in '
      '${Directory.current.path}',
    );
    exit(1);
  }

  final svgFile = File('doc/platform.svg')..writeAsStringSync(_buildSvg());
  stdout.writeln('wrote ${svgFile.path}');

  final rsvg = _which('rsvg-convert');
  if (rsvg == null) {
    stdout.writeln(
      'rsvg-convert not found; install it (brew install librsvg) or run:\n'
      '  rsvg-convert -z $_scale doc/platform.svg -o doc/platform.png',
    );
    return;
  }
  final result = Process.runSync(rsvg, [
    '-z',
    '$_scale',
    'doc/platform.svg',
    '-o',
    'doc/platform.png',
  ]);
  if (result.exitCode != 0) {
    stderr
      ..writeln('rsvg-convert failed with exit code ${result.exitCode}')
      ..writeln((result.stderr as String).trim());
    exit(result.exitCode);
  }
  _quantize(File('doc/platform.png'));
  stdout.writeln(
    'wrote doc/platform.png '
    '(${(_width * _scale).round()}x${(_height * _scale).round()}, '
    '${File('doc/platform.png').lengthSync()} bytes)',
  );
}

/// Reduces [png] to a 64-colour palette in place, when ImageMagick is around.
///
/// The chart is flat fills and antialiased text, which needs nowhere near
/// truecolour. Quantizing keeps the full resolution and takes the file to
/// roughly a third of its size; downscaling instead would cost legibility and
/// often lands larger, because interpolation invents colours.
void _quantize(File png) {
  final magick = _which('magick') ?? _which('convert');
  if (magick == null) {
    stdout.writeln(
      'ImageMagick not found; skipping the palette pass '
      '(brew install imagemagick to get the smaller file)',
    );
    return;
  }
  final before = png.lengthSync();
  final result = Process.runSync(magick, [png.path, '-colors', '64', png.path]);
  if (result.exitCode != 0) {
    stderr
      ..writeln('$magick failed with exit code ${result.exitCode}')
      ..writeln((result.stderr as String).trim());
    exit(result.exitCode);
  }
  final after = png.lengthSync();
  stdout.writeln(
    'palette pass: $before -> $after bytes '
    '(${(100 * after / before).round()}%)',
  );
}

/// Horizontal position of [ratio] on the log axis.
double _x(double ratio) {
  final span = math.log(_domainMax / _domainMin);
  final at = math.log(ratio / _domainMin);
  return _plotLeft + (at / span) * (_plotRight - _plotLeft);
}

String _buildSvg() {
  final b = StringBuffer()
    ..writeln(
      '<svg xmlns="http://www.w3.org/2000/svg" '
      'width="$_width" height="$_height" '
      'viewBox="0 0 $_width $_height" font-family="$_mono">',
    )
    ..writeln('<rect width="$_width" height="$_height" fill="$_paper"/>')
    ..writeln(_text(44, 48, _title, size: 21, fill: _ink, weight: 'bold'))
    ..writeln(_text(44, 76, _subtitleA, size: 12, fill: _body))
    ..writeln(_text(44, 95, _subtitleB, size: 12, fill: _body));

  _writeLegend(b);
  _writeAxis(b);

  for (var i = 0; i < _targets.length; i++) {
    _writeGroup(b, _targets[i], _plotTop + i * _groupPitch);
  }

  for (var i = 0; i < _footer.length; i++) {
    b.writeln(_text(44, 546 + i * 19, _footer[i], size: 11.5, fill: _muted));
  }
  b
    ..writeln(
      _text(
        44,
        620,
        'test/platform_cost_test.dart  ·  Apple M-series  ·  Dart 3.11  ·  '
        'method in doc/web-performance.md',
        size: 10.5,
        fill: _grid,
      ),
    )
    ..writeln('</svg>');
  return b.toString();
}

void _writeLegend(StringBuffer b) {
  const y = 126.0;
  b
    ..writeln(_swatch(44, y - 9, _amber))
    ..writeln(_text(62, y, _legendEmulated, size: 12, fill: _amber))
    ..writeln(_swatch(330, y - 9, _sky))
    ..writeln(_text(348, y, _legendConditional, size: 12, fill: _sky));
}

void _writeAxis(StringBuffer b) {
  for (final tick in _ticks) {
    final x = _x(tick);
    final isReference = tick == 1.0;
    b
      ..writeln(
        '<line x1="${_f(x)}" y1="${_f(_plotTop - 10)}" '
        'x2="${_f(x)}" y2="${_f(_axisY)}" '
        'stroke="${isReference ? _rule : _grid}" stroke-width="1"'
        '${isReference ? '' : ' stroke-dasharray="2 4"'}/>',
      )
      ..writeln(
        _text(
          x,
          _axisY + 20,
          _tickLabel(tick),
          size: 11.5,
          fill: isReference ? _body : _muted,
          anchor: 'middle',
        ),
      );
  }
  b
    ..writeln(_text(_plotLeft, _axisY + 44, 'faster', size: 11.5, fill: _muted))
    ..writeln(
      _text(
        _plotRight,
        _axisY + 44,
        'slower',
        size: 11.5,
        fill: _muted,
        anchor: 'end',
      ),
    );
}

String _tickLabel(double tick) => tick < 1 ? '${tick}x' : '${tick.round()}x';

void _writeGroup(StringBuffer b, _Target target, double top) {
  b
    ..writeln(
      _text(
        44,
        top + 29,
        '${target.name}, ${target.detail}',
        size: 14,
        fill: _ink,
      ),
    )
    ..writeln(
      _text(
        44,
        top + 50,
        'hand-written loop ${_us(target.handWritten)}',
        size: 11.5,
        fill: _muted,
      ),
    );

  _writeBar(
    b,
    y: top + 8,
    micros: target.emulated,
    ratio: target.emulated / target.handWritten,
    color: _amber,
  );
  _writeBar(
    b,
    y: top + 8 + _barHeight + _rowGap,
    micros: target.conditional,
    ratio: target.conditional / target.handWritten,
    color: _sky,
  );
}

void _writeBar(
  StringBuffer b, {
  required double y,
  required int micros,
  required double ratio,
  required String color,
}) {
  final origin = _x(1.0);
  final end = _x(ratio);
  final left = math.min(origin, end);
  final width = (origin - end).abs();
  final centre = y + _barHeight / 2 + 4;

  b.writeln(
    '<rect x="${_f(left)}" y="${_f(y)}" width="${_f(math.max(width, 5))}" '
    'height="$_barHeight" fill="$color" rx="2"/>',
  );

  // Wide bars carry their microsecond figure inside, against the fill; narrow
  // ones would clip it, so those put everything outside the bar end.
  final label = '${_ratio(ratio)}x';
  if (width >= 74) {
    b.writeln(
      _text(
        left + 10,
        centre,
        _us(micros),
        size: 12,
        fill: _paper,
        weight: 'bold',
      ),
    );
    b.writeln(
      end < origin
          ? _text(end - 10, centre, label, size: 12, fill: color, anchor: 'end')
          : _text(end + 10, centre, label, size: 12, fill: color),
    );
  } else {
    final text = '$label  ·  ${_us(micros)}';
    b.writeln(
      end < origin
          ? _text(end - 10, centre, text, size: 12, fill: color, anchor: 'end')
          : _text(end + 10, centre, text, size: 12, fill: color),
    );
  }
}

String _us(int micros) => '${_thousands(micros)} us';

String _thousands(int n) {
  final digits = n.toString();
  final out = StringBuffer();
  for (var i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) out.write(',');
    out.write(digits[i]);
  }
  return out.toString();
}

/// One decimal past ten, two below it.
///
/// Truncating 18.53 to "18" while rounding 44.49 to "45" is how a chart and
/// the prose beside it start disagreeing, which is what happened to
/// `doc/web-performance.md` before this generator existed.
String _ratio(double ratio) =>
    ratio >= 10 ? ratio.toStringAsFixed(1) : ratio.toStringAsFixed(2);

String _swatch(double x, double y, String fill) =>
    '<rect x="${_f(x)}" y="${_f(y)}" width="12" height="12" fill="$fill" '
    'rx="2"/>';

String _text(
  double x,
  double y,
  String value, {
  required double size,
  required String fill,
  String? anchor,
  String? weight,
}) {
  final a = anchor == null ? '' : ' text-anchor="$anchor"';
  final w = weight == null ? '' : ' font-weight="$weight"';
  return '<text x="${_f(x)}" y="${_f(y)}" font-size="$size" fill="$fill"$a$w>'
      '${_escape(value)}</text>';
}

String _escape(String value) => value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');

String _f(double value) => value.toStringAsFixed(1);

String? _which(String executable) {
  final result = Process.runSync('which', [executable]);
  if (result.exitCode != 0) return null;
  final path = (result.stdout as String).trim();
  return path.isEmpty ? null : path;
}

class _Target {
  const _Target({
    required this.name,
    required this.detail,
    required this.emulated,
    required this.conditional,
    required this.handWritten,
  });

  /// Runtime the row is about, as a reader would name it.
  final String name;

  /// Compiler or mode, printed under [name].
  final String detail;

  /// Microseconds per query with `Float32x4` kernels everywhere.
  final int emulated;

  /// Microseconds per query with the kernels 1.1.0 selects for this target.
  final int conditional;

  /// Microseconds per query for the same search written without the package.
  final int handWritten;
}
