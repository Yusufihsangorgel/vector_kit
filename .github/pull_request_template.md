Issue: #<number>

Checklist, matching CI. Run in the repository root:

- [ ] `dart pub get`
- [ ] `dart format --output=none --set-exit-if-changed .`
- [ ] `dart analyze --fatal-infos`
- [ ] `dart test`
- [ ] `dart test -p chrome`
- [ ] `dart test -p chrome -c dart2wasm`
- [ ] `dart compile wasm example/semantic_search.dart -o /tmp/vk.wasm`
- [ ] `CHANGELOG.md` entry added
