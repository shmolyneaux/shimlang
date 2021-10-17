# shimlang
A programming language for small, easily-deployable scripts.

This project is currently built with Rust version `cargo 1.57.0-nightly (d56b42c54 2021-09-27)`.

To try it out, run `cargo build` to create the binary `target/debug/shimlang`.

Fork friendly! Instructions will be available for customizing the language and
built-in libraries once there's enough of a language in place to warrant that.

## Easy Distribution

A key goal of the Shimlang project is to provide a portable scripting language
that is _exceptionally_ easy to distribute. That means several things:

- As much as possible (for each platform), binary releases should be static binaries (no dynamic linking to libc)
- Binaries should be small, ideally a few hundred kB (there shouldn't be big concerns about including the binary in a git repo)
    - For the extra-cautious, a bootstrap executable should be available that downloads a specified version
- Scripts should be easily packaged with the interpreter for single-file distribution
- The language should provide familiar syntax. The syntax is largely inspired by Rust, and I haven't seen
  huge complaints about Rust syntax (in constrast with Python, Lisp, Haskell, bash).

When releasing scripts

## Creating a Release

Releases of Shimlang are intended to be very small. The normal `cargo build --release`
process doesn't create binaries that are small enough for this project, and
doesn't create a static binary.

The current release process can be seen by looking at the `.drone.yml` file. As
of writing this, the current Linux binary release process is:
```
cargo +nightly build -p shimlang --release -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort --target x86_64-unknown-linux-musl
strip target/x86_64-unknown-linux-musl/release/shimlang
```

Shimlang can also be compiled to WASM using the wrapper provided by `shimlang-wasm`,
but this is currently experimental.

Instructions for other platforms will come when support for them is available.

## Why Rust?

There are a handful of languages that can support the goals of this project.
Namely:
    - C
    - C++
    - D
    - Rust
    - Zig

Other languages include runtimes that are too big for the goals of this project.

Between those, I would always prefer Zig over C, and I have no exposure to D.
After getting a fair distance with a Zig implementation, I realized how important
RAII would be when dealing with error cases. This left C++ and Rust.

While C++ generally makes low-level memory operations easier than Rust, the
lack of Algebraic Data Types and poor tooling in C++ led to Rust being the
best choice.

There are some significant downsides with Rust though. The primary issue is
opting-out of global allocators and global out-of-memory handlers which greatly
increase the size of the binary. The effort to support Rust for the Linux kernel
will help make this more viable in the future. Unfortunately for now, several
foundational collections needed to be re-implemented to support falliable
allocations. Once the standard library supports fallible allocators, this will
be less of a burden (it wasn't _huge_ but it was a big downside).
