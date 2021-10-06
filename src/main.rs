#![feature(allocator_api)]
#![no_main]

use std::alloc::Allocator;
use std::alloc::Layout;
use std::ptr::NonNull;

use std::ffi::CStr;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::FromRawFd;

#[link(name = "c")]
extern "C" {
    pub fn malloc(size: usize) -> *mut u8;
    pub fn free(ptr: *mut u8);
}

struct StephenAllocator {}

unsafe impl Allocator for StephenAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        // It's not entirely clear what to do about the alignment. I assume we
        // can just assume malloc does the right thing?

        let size = layout.size();
        let ptr: *mut u8 = unsafe{malloc(size)};

        if ptr.is_null() {
            Err(std::alloc::AllocError)
        } else {
            NonNull::new(std::ptr::slice_from_raw_parts_mut(ptr, size)).ok_or(std::alloc::AllocError)
        }
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) {
        free(ptr.as_ptr())
    }
}

fn stdout() -> File {
    unsafe { File::from_raw_fd(1) }
}

#[no_mangle]
pub fn main(_argc: i32, _argv: *const *const i8) {
    let mut vec: Vec<&[u8], StephenAllocator> = Vec::new_in(StephenAllocator {});
    let test = b"hello";
    vec.push(unsafe { CStr::from_ptr(*_argv).to_bytes() });
    vec.push(test);

    let mut stdout = stdout();
    stdout.write(vec[0]).unwrap();
}
