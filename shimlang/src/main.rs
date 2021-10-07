#![feature(allocator_api)]
#![no_main]

use std::alloc::Allocator;
use std::alloc::Layout;
use std::ptr::NonNull;

use std::ffi::CStr;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::FromRawFd;

use acollections::ABox;
use libshim;

#[link(name = "c")]
extern "C" {
    pub fn malloc(size: usize) -> *mut u8;
    pub fn free(ptr: *mut u8);
}

#[derive(Copy, Clone)]
struct StephenAllocator {}

unsafe impl Allocator for StephenAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        // It's not entirely clear what to do about the alignment. I assume we
        // can just assume malloc does the right thing?

        let size = layout.size();
        let ptr: *mut u8 = unsafe { malloc(size) };

        if ptr.is_null() {
            Err(std::alloc::AllocError)
        } else {
            NonNull::new(std::ptr::slice_from_raw_parts_mut(ptr, size))
                .ok_or(std::alloc::AllocError)
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
pub fn main(argc: i32, _argv: *const *const i8) -> Result<(), std::alloc::AllocError> {
    let mut stdout = stdout();

    // TODO: implement a REPL
    if argc != 2 {
        stdout
            .write(b"Expected a single script as an argument\n")
            .unwrap();
        return Ok(());
    }

    let allocator = StephenAllocator {};
    let my_box: ABox<u8, _> = ABox::new(b'A', allocator)?;
    let script_name = unsafe { CStr::from_ptr(*_argv.offset(1)).to_bytes() };

    stdout.write(b"Reading ").unwrap();
    stdout.write(script_name).unwrap();
    stdout.write(b"\n").unwrap();

    let mut file = File::open(&std::str::from_utf8(script_name).unwrap()).unwrap();

    let buf_layout = Layout::array::<u8>(100).map_err(|_| std::alloc::AllocError)?;
    let buf: NonNull<[u8]> = allocator.allocate(buf_layout)?;

    let count = unsafe { file.read(&mut *buf.as_ptr()).unwrap() };
    unsafe { stdout.write(&(*buf.as_ptr())[..count]).unwrap() };
    stdout.write(b"\n").unwrap();

    stdout.write(b"After ").unwrap();
    stdout.write(&[*my_box]).unwrap();
    stdout.write(b" is ").unwrap();
    stdout.write(&[libshim::add_one(b'A')]).unwrap();
    stdout.write(b"\n").unwrap();

    unsafe { allocator.deallocate(buf.cast(), buf_layout) };

    Ok(())
}
