# Atom Table for Rust Changelog

<!--
Copyright 2023, Collabora, Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

## 1.1.0

### Major changes

- Relicense to "MIT or Apache-2.0" to match the rest of the Rust ecosystem.
- Make the argument to `get_or_create_id` generic using `ToOwned` trait, for
  easier argument passing, especially with strings.

### Minor changes

- Add "xtask" setup for easy running of test code coverage.
- Add tests to increase line coverage to 100%
- Add reference to the docs.rs docs in Cargo.toml
- Add more derived traits to `NonUniqueTransformOutputError` for the sake of the
  tests.
- Add some `#[must_use]`.
- Remove leftover redundant assert.
- Improve docs.

## 1.0.0

- Initial release
