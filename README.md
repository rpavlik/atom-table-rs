# Atom Table for Rust

<!--
Copyright 2023, Collabora, Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->
[![Crates.io](https://img.shields.io/crates/v/atom_table)](https://crates.io/crates/atom_table)
[![docs.rs](https://img.shields.io/docsrs/atom_table)](https://docs.rs/atom_table/latest)
[![REUSE status](https://api.reuse.software/badge/github.com/rpavlik/atom-table-rs)](https://api.reuse.software/info/github.com/rpavlik/atom-table-rs)
[![codecov](https://codecov.io/github/rpavlik/atom-table-rs/branch/main/graph/badge.svg?token=0C2IIDKAT7)](https://codecov.io/github/rpavlik/atom-table-rs)

A simple data structure, allowing you to refer to "hard to handle" values by
"easy to handle" IDs, with lookups in both directions. I imagine such a crate
already exists but using a name I do not know, so here is mine anyway.

## Development and contribution

I try to keep the test coverage high because this is a simple data structure and
it should be possible to easily exercise it all.

To generate a local test coverage report as HTML:

```sh
cargo xtask coverage --dev
```

To generate the coverage in lcov format for (presumably) your editor to use:

```sh
cargo xtask coverage
```

To run most of the CI checks that will take place:

```sh
cargo xtask ci
```

## License

Licensed under either of the
[Apache License, Version 2.0](LICENSES/Apache-2.0.txt) or the
[MIT license](LICENSES/MIT.txt) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

This software conforms to the [REUSE specification](https://reuse.software).
