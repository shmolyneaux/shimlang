#![feature(allocator_api)]
#![no_main]

use std::alloc::Allocator;
use std::alloc::Layout;
use std::ptr::NonNull;

use libshim;
use libshim::Printer;

extern {
    // Lol, let's give JS a pointer and a len and have it read the memory directly
    // through get_memory_byte
    pub fn js_print(offset: usize, count: usize);
}

// The next location to allocate, already 8-byte-aligned
static mut BUMP_ALLOC_NEXT: usize = 0;
static mut BUMP_ALLOC_TOTAL: usize = 0;
const BYTES_PER_PAGE: usize = 65536;

#[derive(Debug, Copy, Clone)]
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

struct WasmPrinter {
    allocator: BumpAllocator
}

impl libshim::Printer for WasmPrinter {
    fn print(&mut self, text: &[u8]) {
        let len = text.len();
        let layout = Layout::array::<u8>(len).unwrap();

        // Allocate memory for writing a message using our bump allocator. It
        // won't get freed until the next script is run.
        let print_ptr: *mut u8 = self.allocator.allocate(layout).unwrap().as_ptr().cast();

        let print_str: &mut [u8] = unsafe {
            &mut *std::ptr::slice_from_raw_parts_mut(print_ptr, len)
        };

        print_str.copy_from_slice(text);

        unsafe { js_print(print_ptr as usize, len) };
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
            let mut printer = WasmPrinter {allocator};
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

// If I can't figure out how to write directly to memory, I'm just going to
// write out the script. Byte. By. Byte.
#[no_mangle]
pub extern fn set_file_byte(idx: isize, val: u8) {
    unsafe {
        *FILE_START.offset(idx) = val;
    }
}

#[no_mangle]
pub extern fn get_memory_byte(ptr: usize) -> u8 {
    unsafe {
        *(ptr as *mut u8)
    }
}


#[no_mangle]
pub extern fn run_file() {
    if run_file_inner().is_err() {
        let allocator = BumpAllocator {};
        let mut printer = WasmPrinter {allocator};
        printer.print(b"alloc error");
    }
}

fn run_file_inner() -> Result<(), std::alloc::AllocError> {
    let allocator = BumpAllocator {};
    let mut printer = WasmPrinter {allocator};

    let mut interpreter = libshim::Interpreter::new(allocator);

    let file_contents = unsafe {
        std::ptr::slice_from_raw_parts_mut(FILE_START, FILE_SIZE)
    };

    interpreter.set_print_fn(&mut printer);
    match interpreter.interpret(unsafe { &*file_contents }) {
        Err(_) => printer.print(b"Error while running script"),
        _ => {},
    }

    Ok(())
}
