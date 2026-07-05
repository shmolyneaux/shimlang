#![allow(dead_code)]

#[cfg(feature = "facet")]
use facet::Facet;

// No-op profiling zone used when the `tracy` feature is disabled. With `tracy`
// enabled the real `zone_scoped!` macro comes from the `shm-tracy` crate
// instead (see the `use shm_tracy::*` imports in the instrumented modules).
#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! zone_scoped {
    ($($arg:tt)*) => {{}};
}

pub mod compile;
pub mod lex;
pub mod parse;
#[macro_use]
pub mod mem;
pub mod runtime;
pub mod shimlibs;

pub use compile::*;
pub use lex::*;
pub use mem::*;
pub use parse::*;
pub use runtime::*;
