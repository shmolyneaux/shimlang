# shimlang
A programming language for small, easily-deployable scripts.

This project is currently built with Zig version `0.9.0-dev.1076+d2b5105f5`.

Run `zig build` to create the binary `zig-out/bin/shimlang`.

Fork friendly! Instructions will be available for customizing the language and
built-in libraries once there's enough of a language in place to warrant that.

## Maybe Rust?

TBD: This may move to Rust. A big reason for using Zig was good cross-platform
building, small binaries, static compilation, explicit (fallible) allocations,
and low-level memory management.

However, it seems like using (or building up) an ecosystem for fallible allocators
in Rust is pretty reasonable. I originally stayed away since it seemed like it
would be difficult to get small static binaries without `no_std`, but it ended
up being reasonable to get to a 5 kB "hello world" in a static binary.

There are limitations here though. It's easy to pull in things from the standard
library that balloon the binary size. Even there, it seems like `bloaty` does
a better job with Rust binaries, and Rust binaries seem to have fewer strings
left in the binary compared with Zig.

I expect it will be harder to do things like intrusive linked lists. On the other
hand, performance is not a primary goal, so alternative implementations aren't
a big deal, even if they're not the most efficient. The upside here is that Rust
has "drop" and "nodrop", and the ownership system helps prevent the sorts of
mistakes that cause leaks in Zig. The downside is that the Zig General Purpose
Allocator is _awesome_ for finding memory leaks, and Rust doesn't seem to have
an equivalent for tracing allocations, which is important since I'm going to
need to delve into `unsafe` land for implementing a lot of the allocator
collections that Zig provides directly.

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
