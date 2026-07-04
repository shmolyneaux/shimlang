use core::ffi::{c_uint, c_char, c_int};


#[repr(C)]
#[derive(Copy, Clone)]
pub struct TracyCZoneCtx {
    pub id: u32,
    pub active: c_int,
}

#[repr(C)]
pub struct ___tracy_source_location_data {
    pub name: *const c_char,
    pub function: *const c_char,
    pub file: *const c_char,
    pub line: u32,
    pub color: u32,
}

unsafe impl Sync for ___tracy_source_location_data {}

pub struct TracyZone {
    pub ctx: TracyCZoneCtx
}

impl Drop for TracyZone {
    fn drop(&mut self) {
        unsafe { tracy_zone_end(self.ctx) };
    }
}

#[macro_export]
macro_rules! zone_scoped {
    ($name:expr) => {{
        static LOC: ___tracy_source_location_data = ___tracy_source_location_data {
            name: concat!($name, "\0").as_ptr() as *const i8,
            function: c"Rust Function".as_ptr() as *const i8,
            file: concat!(env!("CARGO_MANIFEST_DIR"), "\\", file!(), "\0").as_ptr() as *const i8,
            line: line!(),
            color: 0,
        };
        unsafe {
            $crate::TracyZone { ctx: $crate::___tracy_emit_zone_begin(&LOC as *const ___tracy_source_location_data, 1) }
        }
    }};
}

#[cfg(all(feature = "tracy-enable", not(test)))]
unsafe extern "C" {
    pub fn tracy_zone_begin_n(name: *const c_char, active: c_int) -> TracyCZoneCtx;
    pub fn tracy_zone_begin_ns(name: *const c_char, depth: c_int, active: c_int) -> TracyCZoneCtx;
    pub fn tracy_zone_end(ctx: TracyCZoneCtx);
    pub fn tracy_zone_text(ctx: TracyCZoneCtx, txt: *const c_char, len: c_uint);
    pub fn tracy_zone_name(ctx: TracyCZoneCtx, txt: *const c_char, len: c_uint);
    pub fn tracy_zone_color(ctx: TracyCZoneCtx, color: c_uint);

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn ___tracy_emit_zone_begin(loc: *const ___tracy_source_location_data, active: i32) -> TracyCZoneCtx;
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn ___tracy_emit_zone_end(ctx: TracyCZoneCtx);
}

#[cfg(all(not(target_arch = "x86_64"), not(test)))]
pub unsafe fn ___tracy_emit_zone_begin(_loc: *const ___tracy_source_location_data, _active: i32) -> TracyCZoneCtx { TracyCZoneCtx {id: 0, active: 0} }
#[cfg(all(not(target_arch = "x86_64"), not(test)))]
pub unsafe fn ___tracy_emit_zone_end(_ctx: TracyCZoneCtx) {}

#[cfg(any(not(feature = "tracy-enable"), test))]
mod stubs {
    use crate::*;
    pub unsafe fn tracy_zone_begin_n(_name: *const c_char, _active: c_int) -> TracyCZoneCtx { TracyCZoneCtx {id: 0, active: 0} }
    pub unsafe fn tracy_zone_begin_ns(_name: *const c_char, _depth: c_int, _active: c_int) -> TracyCZoneCtx { TracyCZoneCtx {id: 0, active: 0} }
    pub unsafe fn tracy_zone_end(_ctx: TracyCZoneCtx) {}
    pub unsafe fn tracy_zone_text(_ctx: TracyCZoneCtx, _txt: *const c_char, _len: c_uint) {}
    pub unsafe fn tracy_zone_name(_ctx: TracyCZoneCtx, _txt: *const c_char, _len: c_uint) {}
    pub unsafe fn tracy_zone_color(_ctx: TracyCZoneCtx, _color: c_uint) {}

    pub unsafe fn ___tracy_emit_zone_begin(_loc: *const ___tracy_source_location_data, _active: i32) -> TracyCZoneCtx { TracyCZoneCtx {id: 0, active: 0} }
    pub unsafe fn ___tracy_emit_zone_end(_ctx: TracyCZoneCtx) {}
}

#[cfg(any(not(feature = "tracy-enable"), test))]
pub use stubs::*;