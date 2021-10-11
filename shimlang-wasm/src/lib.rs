#![feature(allocator_api)]
#![no_main]

use std::alloc::Allocator;
use std::alloc::Layout;
use std::ptr::NonNull;

use libshim;
use libshim::Printer;

extern {
    // For now... print a single byte at a time so that we don't need to do
    // anything funky with returning references.
    pub fn js_print(byte: u8);
}

// The next location to allocate, already 8-byte-aligned
static mut BUMP_ALLOC_NEXT: usize = 0;
static mut BUMP_ALLOC_TOTAL: usize = 0;
const BYTES_PER_PAGE: usize = 65536;

#[derive(Copy, Clone)]
struct BumpAllocator {}
impl libshim::Allocator for BumpAllocator {}

impl BumpAllocator {
    fn reserve(&self, bytes: usize) -> Result<(), std::alloc::AllocError> {
        let new_page_count = if bytes % BYTES_PER_PAGE == 0 {
            bytes / BYTES_PER_PAGE
        } else {
            1 + bytes / BYTES_PER_PAGE
        };

        // This returns the last page number of the previous allocation.
        let last_page = core::arch::wasm32::memory_grow(0, new_page_count);

        // The return value is usize::MAX when there's an error
        if last_page == usize::MAX {
            Err(std::alloc::AllocError)
        } else {
            unsafe { BUMP_ALLOC_TOTAL += new_page_count * BYTES_PER_PAGE };
            Ok(())
        }
    }

    fn clear(&self) {
        unsafe { BUMP_ALLOC_NEXT = 0 };
    }
}

unsafe impl Allocator for BumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        // We ignore alignment, hopefully 8-bytes is fine...
        let size = layout.size();

        let remaining: usize = unsafe { BUMP_ALLOC_TOTAL - BUMP_ALLOC_NEXT };
        if remaining < size {
            let bytes_to_allocate = size - remaining;
            self.reserve(bytes_to_allocate)?;
        }

        // Never return 0x0
        unsafe {
            if BUMP_ALLOC_NEXT == 0 {
                // It is incorrect to do this with how we allocate new pages above,
                // but it feels slightly less bad than returning a pointer to
                // address 0x0
                BUMP_ALLOC_NEXT += 8;
            }
        }

        let ptr = unsafe { BUMP_ALLOC_NEXT as *mut u8 };

        // Bump the pointer, keeping it 8-byte-aligned;
        unsafe { BUMP_ALLOC_NEXT += size + size % 8 };

        Ok(NonNull::new(std::ptr::slice_from_raw_parts_mut(ptr, size)).unwrap())
    }
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
        // Do nothing! This is a bump allocator
    }
}

struct WasmPrinter {}

impl libshim::Printer for WasmPrinter {
    fn print(&mut self, text: &[u8]) {
        for c in text {
            unsafe { js_print(*c) };
        }
    }
}

static mut FILE_START: *mut u8 = 0 as *mut u8;
static mut FILE_SIZE: usize = 0;

#[no_mangle]
pub extern fn clear_memory_and_allocate_file(size: usize) -> usize {
    let allocator = BumpAllocator {};
    allocator.clear();

    let layout = match Layout::array::<u8>(size) {
        Ok(l) => l,
        Err(_) => {
            let mut printer = WasmPrinter {};
            printer.print(b"alloc error");

            return 0;
        }
    };

    let ptr: *mut u8 = match allocator.allocate(layout) {
        Ok(ptr) => ptr.as_ptr().cast(),
        Err(_) => return 0,
    };

    unsafe {
        FILE_START = ptr;
        FILE_SIZE = size;
    }

    ptr as usize
}


#[no_mangle]
pub extern fn run_file() {
    if run_file_inner().is_err() {
        let mut printer = WasmPrinter {};
        printer.print(b"alloc error");
    }
}

fn run_file_inner() -> Result<(), std::alloc::AllocError> {
    let mut printer = WasmPrinter {};
    let allocator = BumpAllocator {};

    let mut interpreter = libshim::Interpreter::new(allocator);

    let file_contents = unsafe {
        std::ptr::slice_from_raw_parts_mut(FILE_START, FILE_SIZE)
    };

    interpreter.set_print_fn(&mut printer);
    interpreter.interpret(unsafe { &*file_contents }).unwrap();

    Ok(())
}
