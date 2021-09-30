# shimlang
A programming language for small, easily-deployable scripts.

This project is currently built with Zig version `0.9.0-dev.1076+d2b5105f5`.

Run `zig build` to create the binary `zig-out/bin/shimlang`.

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

To build small releases, update `build.zig` to set `strip = true` and run the
following build command:
```
zig build -Drelease-small
```

To analyze code size, use [bloaty](https://github.com/google/bloaty). This is
what I typically run:
```
bloaty shimlang -d symbols -n 50 -s file -C none
```
