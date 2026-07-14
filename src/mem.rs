use crate::runtime::*;
use crate::shimlibs::*;
#[cfg(feature = "tracy")]
use shm_tracy::*;
use std::any::TypeId;
use std::collections::{HashMap, BTreeMap};
use std::ops::Range;
use std::ops::{Add, AddAssign, Sub, SubAssign};

#[cfg(feature = "gc_debug")]
use crate::lex::debug_u8s;

#[derive(Debug)]
pub struct Config {
    // There are max 2^24 addressable values, each 8 bytes large
    // This value can be up to 2^27-1.
    pub memory_space_bytes: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory_space_bytes: MAX_U24 * 8,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialOrd, Ord, Copy, Clone, PartialEq)]
#[repr(Rust, packed)]
pub struct u24(pub(crate) [u8; 3]);

impl std::fmt::Debug for u24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "u24({})", u32::from(*self))
    }
}
pub(crate) const MAX_U24: u32 = 0xFFFFFF;
pub(crate) const PAGE_SIZE: usize = 64;

impl From<usize> for u24 {
    fn from(val: usize) -> Self {
        (val as u32).into()
    }
}

impl From<i32> for u24 {
    fn from(val: i32) -> Self {
        (val as u32).into()
    }
}

impl From<u32> for u24 {
    fn from(val: u32) -> Self {
        let b = val.to_be_bytes();
        u24([b[1], b[2], b[3]])
    }
}

impl From<u64> for u24 {
    fn from(val: u64) -> Self {
        (val as u32).into()
    }
}

impl From<u24> for u32 {
    fn from(val: u24) -> u32 {
        u32::from_be_bytes([0, val.0[0], val.0[1], val.0[2]])
    }
}

impl From<u24> for usize {
    fn from(val: u24) -> usize {
        u32::from(val) as usize
    }
}

impl From<u24> for u64 {
    fn from(val: u24) -> u64 {
        u32::from(val) as u64
    }
}

impl Add<u8> for u24 {
    type Output = u24;

    fn add(self, rhs: u8) -> u24 {
        self + rhs as u32
    }
}

impl Add<i32> for u24 {
    type Output = u24;

    fn add(self, rhs: i32) -> u24 {
        let val = (u32::from(self) as i32 + rhs) as u32;
        val.into()
    }
}

impl Add<u32> for u24 {
    type Output = u24;

    fn add(self, rhs: u32) -> u24 {
        (u32::from(self) + rhs).into()
    }
}

impl Sub<u32> for u24 {
    type Output = u24;

    fn sub(self, rhs: u32) -> u24 {
        (u32::from(self) - rhs).into()
    }
}

impl Add<u24> for u24 {
    type Output = u24;

    fn add(self, rhs: u24) -> u24 {
        (u32::from(self) + u32::from(rhs)).into()
    }
}

impl Sub<u24> for u24 {
    type Output = u24;

    fn sub(self, rhs: u24) -> u24 {
        (u32::from(self) - u32::from(rhs)).into()
    }
}

impl AddAssign<u32> for u24 {
    fn add_assign(&mut self, rhs: u32) {
        *self = (u32::from(*self) + rhs).into()
    }
}

impl SubAssign<u32> for u24 {
    fn sub_assign(&mut self, rhs: u32) {
        *self = (u32::from(*self) - rhs).into()
    }
}

impl AddAssign<u24> for u24 {
    fn add_assign(&mut self, rhs: u24) {
        *self = (u32::from(*self) + u32::from(rhs)).into()
    }
}

impl SubAssign<u24> for u24 {
    fn sub_assign(&mut self, rhs: u24) {
        *self = (u32::from(*self) - u32::from(rhs)).into()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FreeBlock {
    #[cfg(feature = "dev")]
    pub pos: u24,
    #[cfg(feature = "dev")]
    pub size: u24,

    #[cfg(not(feature = "dev"))]
    pos: u24,
    #[cfg(not(feature = "dev"))]
    size: u24,
}

impl FreeBlock {
    fn new(pos: u24, size: u24) -> Self {
        Self { pos, size }
    }

    pub fn end(&self) -> u24 {
        self.pos + self.size
    }
}

/// Type registration entry stored in the MMU for each unique ShimNative type.
/// Allows reconstructing a trait object from raw memory without a Box.
pub struct NativeTypeInfo {
    pub type_id: TypeId,
    /// The vtable pointer extracted from a `&dyn ShimNative` fat pointer for this type.
    pub(crate) vtable: *const (),
    /// Size of the native value in 8-byte words (rounded up).
    pub word_count: usize,
}

// SAFETY: The vtable pointer is valid for the lifetime of the program and is
// not mutated after insertion.
unsafe impl Send for NativeTypeInfo {}
unsafe impl Sync for NativeTypeInfo {}

#[cfg_attr(feature = "facet", derive(Facet))]
pub struct MMU {
    // This is the raw memory managed by the MMU
    mem: Vec<u64>,
    // Dirty bitmask: each bit represents a PAGE_SIZE-word page.
    dirty: Vec<u64>,

    // Block size to location of available blocks
    pub free_blocks: BTreeMap<u32, Vec<u32>>,

    // The memory position beyond the furthest we've allocated
    wilderness: u32,

    // We don't store metadata about any allocations
    // It's up to the caller to know how much memory
    // should be freed.
    /// Maps Rust TypeId to autoincrementing index (stable across uses within a session).
    pub native_type_ids: HashMap<TypeId, u32>,
    /// Maps autoincrementing index back to type info (vtable + word_count).
    /// The index into this Vec is the autoincrementing id stored in ShimValue::Native.
    pub native_type_registry: Vec<NativeTypeInfo>,

    /// The position of ShimValue::Native values in memory that need to be
    /// explicitly dropped in the sweep phase when unreachable by the GC. These
    /// are the types that set needs_drop() to true.
    /// Each entry is (mmu_position, type_idx).
    droppable_native_pos: Vec<(u32, u24)>,

    /// Set to true while gc_drop callbacks are executing. Any MMU allocation
    /// attempted in this state panics, because newly allocated words would be
    /// invisible to the GC bitmask and immediately reclaimed by sweep.
    dropping_native: bool,
}

macro_rules! alloc {
    ($mmu:expr, $count:expr, $msg:expr) => {{
        #[cfg(debug_assertions)]
        {
            //$mmu.alloc_debug($count, $msg)
            $mmu.alloc_no_debug($count)
        }

        #[cfg(not(debug_assertions))]
        {
            $mmu.alloc_no_debug($count)
        }
    }};
}

impl MMU {
    pub fn mem_high_point(&self) -> u32 {
        self.wilderness
    }
    pub(crate) fn with_capacity(word_count: u24) -> Self {
        let mem = vec![0; usize::from(word_count)];
        let free_blocks = BTreeMap::new();
        let wilderness = 1;
        let dirty = vec![0u64; usize::from(word_count).div_ceil(PAGE_SIZE * 64)];
        Self {
            mem,
            dirty,
            free_blocks,
            wilderness,
            native_type_ids: HashMap::new(),
            native_type_registry: Vec::new(),
            droppable_native_pos: Vec::new(),
            dropping_native: false,
        }
    }

    pub fn mem(&self) -> &[u64] {
        &self.mem
    }

    pub fn mem_mut(&mut self, start: usize, len: usize) -> &mut [u64] {
        self.mark_dirty_range(start, start + len);
        &mut self.mem[start..start + len]
    }

    fn mark_dirty_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let first_page = start / PAGE_SIZE;
        let last_page = (end - 1) / PAGE_SIZE;
        for page in first_page..=last_page {
            self.dirty[page / 64] |= 1u64 << (page % 64);
        }
    }

    fn clear_dirty_range(&mut self, start: usize, end: usize) {
        let first_full_page = start.div_ceil(PAGE_SIZE);
        let last_full_page = end / PAGE_SIZE;
        for page in first_full_page..last_full_page {
            self.dirty[page / 64] &= !(1u64 << (page % 64));
        }
    }

    pub unsafe fn get<T: 'static>(&self, word: u24) -> &T {
        unsafe {
            let ptr: *const T = &self.mem[usize::from(word)] as *const u64 as *const T;
            &*ptr
        }
    }

    /// Like `get_mut` but returns a raw pointer so the MMU borrow is released
    /// immediately. Callers that need to reborrow the MMU after obtaining a
    /// mutable reference into it must use this instead of `get_mut`.
    pub fn get_ptr_mut<T>(&mut self, word: u24) -> *mut T {
        let word_usize = usize::from(word);
        self.mark_dirty_range(word_usize, word_usize + std::mem::size_of::<T>().div_ceil(8));
        self.mem.as_mut_ptr().wrapping_add(word_usize) as *mut T
    }

    pub unsafe fn get_mut<T>(&mut self, word: u24) -> &mut T {
        let word_usize = usize::from(word);
        self.mark_dirty_range(word_usize, word_usize + std::mem::size_of::<T>().div_ceil(8));
        unsafe {
            let ptr: *mut T = &mut self.mem[word_usize] as *mut u64 as *mut T;
            &mut *ptr
        }
    }

    pub fn alloc_and_set<T>(&mut self, value: T, _debug_name: &str) -> Result<u24, String> {
        let word_count: u24 = (std::mem::size_of::<T>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, _debug_name)?;
        unsafe {
            let ptr: *mut T = &mut self.mem[usize::from(position)] as *mut u64 as *mut T;
            ptr.write(value);
        }
        Ok(position)
    }

    pub(crate) fn alloc_str_raw(&mut self, contents: &[u8]) -> Result<u24, String> {
        let total_len = contents.len().div_ceil(8);
        let word_count: u24 = total_len.into();
        let position = alloc!(self, word_count, &format!("str `{}`", debug_u8s(contents)))?;

        let bytes: &mut [u8] = unsafe {
            let u64_slice =
                &mut self.mem[usize::from(position)..(usize::from(position) + total_len)];
            std::slice::from_raw_parts_mut(u64_slice.as_mut_ptr() as *mut u8, contents.len())
        };

        for (idx, b) in contents.iter().enumerate() {
            bytes[idx] = *b;
        }

        Ok(position)
    }

    pub fn alloc_debug(&mut self, words: u24, msg: &str) -> Result<u24, String> {
        let result = self.alloc_no_debug(words)?;
        eprintln!(
            "Alloc {} {}: {}",
            usize::from(words),
            msg,
            usize::from(result)
        );
        Ok(result)
    }

    pub fn alloc_no_debug(&mut self, words: u24) -> Result<u24, String> {
        assert!(
            !self.dropping_native,
            "MMU allocation attempted during gc_drop; newly allocated words would be invisible \
             to the GC bitmask and immediately reclaimed by sweep"
        );
        if u32::from(words) == 0u32 {
            return Ok(0.into());
        }

        // Set to Some(size) if we take the last block in a list
        let mut pop_bin_size = None;

        // If the block we got was bigger than we needed, the remaining words
        // need to be pushed to a separate list (pos, size)
        let mut remaining: Option<FreeBlock> = None;

        let words_u32 = u32::from(words);

        let pos = if let Some((&size, blocks)) = self.free_blocks.range_mut(words_u32..).next() {
            let pos = blocks.pop().unwrap();
            if blocks.is_empty() {
                pop_bin_size = Some(size);
            }
            if size > words_u32 {
                let leftover = size - words_u32;
                remaining = Some(FreeBlock {pos: (pos + words_u32).into(), size: leftover.into()});
            }
            pos
        } else {
            if words_u32 <= (self.mem.len() as u32 - self.wilderness) {
                let pos = self.wilderness;
                self.wilderness += words_u32;
                pos
            } else {
                return Err(format!(
                    "Out of memory: could not allocate {} words (heap exhausted)",
                    words_u32
                ));
            }
        };

        if let Some(size) = pop_bin_size {
            self.free_blocks.remove(&size);
        }

        if let Some(FreeBlock {pos, size}) = remaining {
            self.free_blocks.entry(u32::from(size)).or_default().push(u32::from(pos));
        }

        self.mark_dirty_range(pos as usize, pos as usize + usize::from(words));

        return Ok(u24::from(pos));
    }

    /**
     * Returns the position in `self.mem` of the block allocated
     */
    #[allow(dead_code)]
    fn alloc(&mut self, size: u24) -> Result<u24, String> {
        self.alloc_debug(size, "Unspecified alloc")
    }

    pub fn free(&mut self, pos: u24, size: u24) {
        if u32::from(pos) == 0 || u32::from(size) == 0 {
            return;
        }

        // If the range is freed it should be unused so the bytes don't need to
        // be copied over even if they were modified
        self.clear_dirty_range(usize::from(pos), usize::from(pos) + usize::from(size));

        self.free_blocks.entry(u32::from(size)).or_default().push(u32::from(pos));
    }

    /** DOES NOT DROP T, but nothing in the MMU should need to drop **/
    pub fn free_obj<T>(&mut self, pos: u24) {
        let word_count: u24 = (std::mem::size_of::<T>() as u32).div_ceil(8).into();
        self.free(pos, word_count);
    }

    // MMU methods that depend on runtime types (ShimValue, ShimDict, ShimList, ShimFn, etc.)

    pub fn alloc_str(&mut self, contents: &[u8]) -> Result<ShimValue, String> {
        if contents.len() > u16::MAX as usize {
            return Err(format!(
                "String length {} exceeds the maximum of {} bytes",
                contents.len(),
                u16::MAX
            ));
        }
        let pos = self.alloc_str_raw(contents)?;
        Ok(ShimValue::String(contents.len() as u16, 0, pos))
    }

    pub fn alloc_dict_raw(&mut self) -> Result<u24, String> {
        let word_count: u24 = (std::mem::size_of::<ShimDict>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, "Dict")?;
        unsafe {
            let ptr: *mut ShimDict = &mut self.mem[usize::from(position)] as *mut u64 as *mut ShimDict;
            ptr.write(ShimDict::new());
        }
        Ok(position)
    }

    pub fn alloc_dict(&mut self) -> Result<ShimValue, String> {
        Ok(ShimValue::Dict(self.alloc_dict_raw()?))
    }

    pub fn alloc_set_raw(&mut self) -> Result<u24, String> {
        let word_count: u24 = (std::mem::size_of::<ShimSet>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, "Set")?;
        unsafe {
            let ptr: *mut ShimSet = &mut self.mem[usize::from(position)] as *mut u64 as *mut ShimSet;
            ptr.write(ShimSet::new());
        }
        Ok(position)
    }

    pub fn alloc_set(&mut self) -> Result<ShimValue, String> {
        Ok(ShimValue::Set(self.alloc_set_raw()?))
    }

    pub fn alloc_tuple(&mut self, items: &[ShimValue]) -> Result<ShimValue, String> {
        let word_count: u24 = items.len().into();
        let position = alloc!(self, word_count, "tuple")?;
        for (idx, val) in items.iter().enumerate() {
            self.mem[usize::from(position)+idx] = val.to_u64();
        }
        Ok(ShimValue::Tuple(word_count, position))
    }

    pub fn alloc_list_raw(&mut self) -> Result<u24, String> {
        let word_count: u24 = (std::mem::size_of::<ShimList>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, "List")?;
        unsafe {
            let ptr: *mut ShimList = &mut self.mem[usize::from(position)] as *mut u64 as *mut ShimList;
            ptr.write(ShimList::new());
        }
        Ok(position)
    }

    pub fn alloc_list(&mut self) -> Result<ShimValue, String> {
        Ok(ShimValue::List(self.alloc_list_raw()?))
    }

    pub fn alloc_fn(&mut self, pc: u32, name: &[u8], captured_scope: u32) -> Result<ShimValue, String> {
        let word_count: u24 = (std::mem::size_of::<ShimFn>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, &format!("Fn `{}`", debug_u8s(name)))?;

        // Allocate the name string
        let name_pos = self.alloc_str_raw(name)?;

        unsafe {
            let ptr: *mut ShimFn = &mut self.mem[usize::from(position)] as *mut u64 as *mut ShimFn;
            ptr.write(ShimFn {
                pc,
                name_len: name.len() as u16,
                name: name_pos,
                captured_scope,
            });
        }
        Ok(ShimValue::Fn(position))
    }

    pub fn alloc_native<T: ShimNative>(&mut self, val: T) -> Result<ShimValue, String> {
        let type_id = TypeId::of::<T>();
        let type_idx = match self.native_type_ids.get(&type_id) {
            Some(&idx) => idx,
            None => {
                let len = self.native_type_registry.len();
                assert!(
                    len < u32::MAX as usize,
                    "native type registry overflow: too many distinct ShimNative types"
                );
                let idx = len as u32;
                // Extract the vtable pointer from a fat reference before moving val.
                let vtable = unsafe {
                    let reference: &dyn ShimNative = &val;
                    let fat_ptr: (*const (), *const ()) = std::mem::transmute(reference);
                    fat_ptr.1
                };
                let word_count = std::mem::size_of::<T>().div_ceil(8);
                self.native_type_ids.insert(type_id, idx);
                self.native_type_registry.push(NativeTypeInfo {
                    type_id,
                    vtable,
                    word_count,
                });
                idx
            }
        };

        let type_idx = u24::from(type_idx);

        // Allocate T directly in MMU memory (no Box).
        let word_count: u24 = (std::mem::size_of::<T>() as u32).div_ceil(8).into();
        let position = alloc!(self, word_count, "Native")?;
        if val.needs_drop() {
            self.droppable_native_pos.push((u32::from(position), type_idx));
        }
        unsafe {
            let ptr: *mut T = &mut self.mem[usize::from(position)] as *mut u64 as *mut T;
            ptr.write(val);
        }
        Ok(ShimValue::Native(type_idx, position))
    }

    pub fn alloc_bound_method(&mut self, obj: &ShimValue, func_pos: u24) -> Result<ShimValue, String> {
        let position = alloc!(self, u24::from(2), "Bound Method")?;
        self.mem[usize::from(position)] = obj.to_u64();
        self.mem[usize::from(position) + 1] = u64::from(func_pos);

        Ok(ShimValue::BoundMethod(position))
    }

    pub fn alloc_bound_native_fn(&mut self, obj: &ShimValue, func: NativeFn) -> Result<ShimValue, String> {
        let position = alloc!(self, 2u32.into(), "Bound Native Fn")?;
        unsafe {
            let obj_ptr: *mut ShimValue =
                &mut self.mem[usize::from(position)] as *mut u64 as *mut ShimValue;
            obj_ptr.write(*obj);
            let fn_ptr: *mut NativeFn =
                &mut self.mem[usize::from(position) + 1] as *mut u64 as *mut NativeFn;
            fn_ptr.write(func);

            Ok(ShimValue::BoundNativeMethod(position))
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemDescriptor {
    pub start: usize,
    pub end: usize,
    pub t: MemDescriptorType,
}

impl MemDescriptor {
    fn other(start: usize, end: usize, text: &str) -> Self {
        Self {
            start,
            end,
            t: MemDescriptorType::Other(text.to_string()),
        }
    }

    fn struct_desc(start: usize, end: usize, type_name: String, members: Vec<ShimValue>) -> Self {
        Self {
            start,
            end,
            t: MemDescriptorType::Struct(type_name, members),
        }
    }

    fn env_header(start: usize, end: usize, text: &str) -> Self {
        Self {
            start,
            end,
            t: MemDescriptorType::EnvHeader(text.to_string()),
        }
    }

    fn env_data(start: usize, end: usize, text: &str) -> Self {
        Self {
            start,
            end,
            t: MemDescriptorType::EnvData(text.to_string()),
        }
    }

    pub fn to_string(&self, mem: &MMU) -> String {
        match &self.t {
            MemDescriptorType::Other(s) => s.clone(),
            MemDescriptorType::EnvHeader(s) => s.clone(),
            MemDescriptorType::EnvData(s) => s.clone(),
            MemDescriptorType::Struct(type_name, members) => {
                let mut s = String::new();
                s.push_str(&format!("{type_name}("));
                for (idx, val) in members.iter().enumerate() {
                    if idx != 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&val.to_string_mem(mem));
                }
                s.push(')');
                s
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum MemDescriptorType {
    Other(String),
    EnvData(String),
    EnvHeader(String),
    Struct(String, Vec<ShimValue>),
}

#[derive(Debug)]
pub struct Bitmask {
    #[cfg(feature = "gc_debug")]
    pub description: HashMap<usize, MemDescriptor>,
    data: Vec<u64>,
}

macro_rules! mark_bit {
    ($mask:expr, $index:expr, $desc:expr) => {{
        #[cfg(feature = "gc_debug")]
        {
            $mask.set($index, &$desc);
        }
        #[cfg(not(feature = "gc_debug"))]
        {
            $mask.set($index);
        }
    }};
}

impl Bitmask {
    pub fn new(num_bits: usize) -> Self {
        // Round up if we don't have a number of bits that's cleanly divisible by 64
        let blocks = num_bits.div_ceil(64);

        Bitmask {
            #[cfg(feature = "gc_debug")]
            description: HashMap::new(),
            data: vec![0; blocks],
        }
    }

    #[cfg(feature = "gc_debug")]
    pub fn set(&mut self, index: usize, description: &MemDescriptor) {
        let (block_idx, bit_offset) = self.pos(index);
        self.data[block_idx] |= 1 << bit_offset;
        self.description.insert(index, description.clone());
    }

    #[cfg(not(feature = "gc_debug"))]
    pub fn set(&mut self, index: usize) {
        let (block_idx, bit_offset) = self.pos(index);
        self.data[block_idx] |= 1 << bit_offset;
    }

    pub fn setx(&mut self, index: usize) {
        let (block_idx, bit_offset) = self.pos(index);
        self.data[block_idx] |= 1 << bit_offset;
    }

    pub fn is_set(&self, index: usize) -> bool {
        let (block_idx, bit_offset) = self.pos(index);
        (self.data[block_idx] & (1 << bit_offset)) != 0
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn find_zeros(&self) -> Vec<Range<usize>> {
        let _zone = zone_scoped!("find_zeros");
        let mut ranges = Vec::new();
        let mut start_of_run: Option<usize> = None;

        for (idx, word) in self.data.iter().enumerate() {
            if *word == 0 {
                if start_of_run.is_none() {
                    start_of_run = Some(idx * 64);
                }
            } else if *word == u64::MAX {
                if let Some(start_bit) = start_of_run {
                    ranges.push(start_bit..(idx * 64));
                    start_of_run = None;
                }
            } else {
                let bit_offset: usize;
                match start_of_run {
                    Some(start_bit) => {
                        bit_offset = word.trailing_zeros() as usize;
                        ranges.push(start_bit..(idx * 64 + bit_offset));
                        start_of_run = None
                    }
                    None => {
                        bit_offset = word.trailing_ones() as usize;
                        start_of_run = Some(idx * 64 + bit_offset);
                    }
                }
                let mut shifted_word = word >> bit_offset;
                for i in bit_offset..64 {
                    let is_zero = (shifted_word & 1) == 0;

                    if is_zero {
                        if start_of_run.is_none() {
                            start_of_run = Some(idx * 64 + i);
                        }
                    } else if let Some(start) = start_of_run {
                        ranges.push(start..idx * 64 + i);
                        start_of_run = None;
                    }
                    shifted_word >>= 1;
                }
            }
        }

        if let Some(start_bit) = start_of_run {
            ranges.push(start_bit..self.data.len() * 64);
        }

        ranges
    }

    fn pos(&self, index: usize) -> (usize, usize) {
        (index / 64, index % 64)
    }
}

pub(crate) struct GC<'a> {
    pub interpreter: &'a mut Interpreter,
    pub mask: Bitmask,
}

impl<'a> GC<'a> {
    pub(crate) fn new(interpreter: &'a mut Interpreter) -> Self {
        let mut mask = Bitmask::new(interpreter.mem.wilderness as usize);
        // Mark word 0 so the GC never frees the sentinel reserved by MMU::with_capacity
        mark_bit!(mask, 0, MemDescriptor::other(0, 1, "null"));
        Self { interpreter, mask }
    }

    pub(crate) fn mark(&mut self, mut vals: Vec<ShimValue>) {
        let _zone = zone_scoped!("GC mark");
        unsafe {
            while !vals.is_empty() {
                match vals.pop().unwrap() {
                    ShimValue::Integer(_)
                    | ShimValue::Float(_)
                    | ShimValue::Bool(_)
                    | ShimValue::Unit
                    | ShimValue::None
                    | ShimValue::StopIteration
                    | ShimValue::Uninitialized => (),
                    ShimValue::Fn(fn_pos) => {
                        let pos: usize = fn_pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let shim_fn_word_count = std::mem::size_of::<ShimFn>().div_ceil(8);
                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(pos, pos + shim_fn_word_count, "ShimFn");
                        for idx in pos..(pos + shim_fn_word_count) {
                            mark_bit!(self.mask, idx, desc);
                        }

                        // Mark the function name string
                        let shim_fn: &ShimFn = self.interpreter.mem.get(fn_pos);
                        vals.push(ShimValue::String(shim_fn.name_len, 0, shim_fn.name));

                        // Mark the captured scope if present
                        if shim_fn.captured_scope != 0 {
                            vals.push(ShimValue::Environment(shim_fn.captured_scope.into()));
                        }
                    }
                    ShimValue::Tuple(len, pos) => {
                        let pos: usize = pos.into();
                        let len: usize = len.into();

                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(pos, pos + len, "Tuple contents");

                        for idx in pos..(pos + len) {
                            vals.push(*self.interpreter.mem.get(idx.into()));
                            mark_bit!(self.mask, idx, desc);
                        }
                    }
                    ShimValue::List(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }

                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(pos, pos + 1, "List header");
                        mark_bit!(self.mask, pos, desc);

                        let lst: &ShimList = self.interpreter.mem.get(pos.into());
                        for idx in 0..lst.len() {
                            vals.push(lst.get(&self.interpreter.mem, idx as isize).unwrap());
                        }

                        let contents_pos = usize::from(lst.data);
                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(
                            contents_pos,
                            contents_pos + lst.capacity(),
                            "List item",
                        );
                        for idx in contents_pos..(contents_pos + lst.capacity()) {
                            mark_bit!(self.mask, idx, desc);
                        }
                    }
                    s @ ShimValue::String(len, offset, pos) => {
                        let pos: usize = usize::from(pos);
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let len = len as usize;
                        let offset = offset as usize;
                        #[cfg(feature = "gc_debug")]
                        let desc = {
                            let contents = s.string_from_mem(&self.interpreter.mem).unwrap();
                            MemDescriptor::other(
                                pos,
                                (offset + len).div_ceil(8),
                                &format!("String: {}", debug_u8s(contents)),
                            )
                        };
                        #[cfg(not(feature = "gc_debug"))]
                        let _ = s; // suppress unused binding warning
                        // TODO: check this...
                        for idx in pos..(pos + (offset + len).div_ceil(8)) {
                            mark_bit!(self.mask, idx, desc);
                        }
                    }
                    ShimValue::Dict(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let dict: &ShimDict = std::mem::transmute(&self.interpreter.mem.mem[pos]);
                        let u64_slice = &self.interpreter.mem.mem[usize::from(dict.entries)
                            ..usize::from(dict.entries) + 3 * (dict.entry_count as usize)];
                        let entries: &[DictEntry] = std::slice::from_raw_parts(
                            u64_slice.as_ptr() as *const DictEntry,
                            u64_slice.len() / 3,
                        );

                        // Push the keys/vals
                        let count: usize = dict.entry_count as usize;
                        for entry in &entries[..count] {
                            if !entry.key.is_uninitialized() {
                                vals.push(entry.key);
                                vals.push(entry.value);
                            }
                        }

                        // Mark the space for the dict struct
                        #[cfg(feature = "gc_debug")]
                        let header_desc = MemDescriptor::other(
                            pos,
                            pos + std::mem::size_of::<ShimDict>().div_ceil(8),
                            "dict header",
                        );
                        for idx in pos..(pos + std::mem::size_of::<ShimDict>().div_ceil(8)) {
                            mark_bit!(self.mask, idx, header_desc);
                        }

                        let size: usize = 1 << dict.size_pow;

                        // Mark the indices array
                        let indices_pos: usize = dict.indices.into();
                        let indices_word_count = if dict.size_pow == 0 {
                            0
                        } else {
                            size.div_ceil(8 / dict.indices_stride_bytes(size))
                        };
                        #[cfg(feature = "gc_debug")]
                        let indices_desc = MemDescriptor::other(
                            indices_pos,
                            indices_pos + indices_word_count,
                            "dict index",
                        );
                        for idx in indices_pos..(indices_pos + indices_word_count) {
                            mark_bit!(self.mask, idx, indices_desc);
                        }

                        // Mark the entries array
                        let entries_pos: usize = dict.entries.into();
                        #[cfg(feature = "gc_debug")]
                        let entries_desc = MemDescriptor::other(
                            entries_pos,
                            entries_pos + dict.capacity() * 3,
                            "dict entries",
                        );
                        for idx in entries_pos..(entries_pos + dict.capacity() * 3) {
                            mark_bit!(self.mask, idx, entries_desc);
                        }
                    }
                    ShimValue::Set(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let set: &ShimSet = std::mem::transmute(&self.interpreter.mem.mem[pos]);

                        // Mark the space for the ShimSet struct
                        #[cfg(feature = "gc_debug")]
                        let header_desc = MemDescriptor::other(
                            pos,
                            pos + std::mem::size_of::<ShimSet>().div_ceil(8),
                            "set header",
                        );
                        for idx in pos..(pos + std::mem::size_of::<ShimSet>().div_ceil(8)) {
                            mark_bit!(self.mask, idx, header_desc);
                        }

                        // And mark the dict as normal. An empty set has no
                        // backing dict allocated yet (dict_pos == 0), so there
                        // is nothing further to mark in that case.
                        if set.dict_pos != u24::from(0) {
                            vals.push(ShimValue::Dict(set.dict_pos));
                        }
                    }
                    ShimValue::StructDef(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def: &StructDef = self.interpreter.mem.get(pos.into());
                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(
                            pos,
                            pos + def.mem_size(),
                            &format!("struct {}", debug_u8s(&def.name)),
                        );
                        for idx in pos..(pos + def.mem_size()) {
                            mark_bit!(self.mask, idx, desc);
                        }

                        // Mark method functions referenced by this struct def
                        for fn_pos in def.method_fn_positions() {
                            vals.push(ShimValue::Fn(fn_pos));
                        }
                    }
                    ShimValue::Struct(def_pos, pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def: &StructDef = self.interpreter.mem.get(def_pos);

                        #[cfg(feature = "gc_debug")]
                        let desc = {
                            let mut members = Vec::new();
                            for idx in pos..(pos + def.member_count as usize) {
                                members.push(ShimValue::from_u64(self.interpreter.mem.mem[idx]));
                            }
                            MemDescriptor::struct_desc(
                                pos,
                                pos + def.member_count as usize,
                                format!("struct {}", debug_u8s(&def.name)),
                                members,
                            )
                        };
                        for idx in pos..(pos + def.member_count as usize) {
                            mark_bit!(self.mask, idx, desc);
                            // Push the members
                            vals.push(ShimValue::from_u64(self.interpreter.mem.mem[idx]));
                        }
                        vals.push(ShimValue::StructDef(def_pos));
                    }
                    ShimValue::NativeFn(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        mark_bit!(
                            self.mask,
                            pos,
                            MemDescriptor::other(pos, pos + 1, "NativeFn")
                        );
                    }
                    ShimValue::Native(type_idx, pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let type_idx: usize = type_idx.into();
                        let info = &self.interpreter.mem.native_type_registry[type_idx];
                        let native_word_count = info.word_count;
                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::other(pos, pos + native_word_count, "Native");
                        for idx in pos..(pos + native_word_count) {
                            mark_bit!(self.mask, idx, desc);
                        }

                        // Reconstruct a fat pointer to call gc_vals() without a Box.
                        let vtable = info.vtable;
                        let data_ptr = &self.interpreter.mem.mem[pos] as *const u64 as *const ();
                        let fat_ptr: (*const (), *const ()) = (data_ptr, vtable);
                        let native_ref: &dyn ShimNative = std::mem::transmute(fat_ptr);
                        vals.extend(native_ref.gc_vals());
                    }
                    ShimValue::BoundMethod(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        // Mark the 2 words used to store the BoundMethod
                        mark_bit!(
                            self.mask,
                            pos,
                            MemDescriptor::other(pos, pos + 1, "Bound Method Obj")
                        );
                        mark_bit!(
                            self.mask,
                            pos + 1,
                            MemDescriptor::other(pos + 1, pos + 2, "Bound Method fn")
                        );

                        let obj = ShimValue::from_u64(self.interpreter.mem.mem[pos]);
                        vals.push(obj);
                        let func_pos = u24::from(self.interpreter.mem.mem[pos + 1]);
                        vals.push(ShimValue::Fn(func_pos));
                    }
                    ShimValue::BoundNativeMethod(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        // Native ShimValue
                        mark_bit!(
                            self.mask,
                            pos,
                            MemDescriptor::other(pos, pos + 2, "BoundNativeMethod")
                        );
                        // Pointer to the fn
                        mark_bit!(
                            self.mask,
                            pos + 1,
                            MemDescriptor::other(pos, pos + 2, "BoundNativeMethod")
                        );

                        // word[pos] is the ShimValue object (e.g. ShimValue::Native); push it to
                        // be processed so its memory is also marked and gc_vals() called on it.
                        let obj = ShimValue::from_u64(self.interpreter.mem.mem[pos]);
                        vals.push(obj);
                    }
                    ShimValue::Environment(pos) => {
                        let og_pos = pos;
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let scope: &EnvScope = self.interpreter.mem.get(og_pos);

                        // Chunk of memory that store the EnvScope metadata
                        #[cfg(feature = "gc_debug")]
                        let desc = MemDescriptor::env_header(
                            pos,
                            pos + std::mem::size_of::<EnvScope>().div_ceil(8),
                            &format!("Envscope header\nparent: {:?}", scope.parent),
                        );
                        for bit in pos..(pos + std::mem::size_of::<EnvScope>().div_ceil(8)) {
                            mark_bit!(self.mask, bit, desc);
                        }

                        // Data block
                        let start = usize::from(scope.data);
                        let end = start + scope.capacity as usize;
                        #[cfg(feature = "gc_debug")]
                        let scope_description =
                            MemDescriptor::env_data(start, end, &scope.to_string(&self.interpreter.mem));
                        for bit in start..end {
                            mark_bit!(self.mask, bit, scope_description);
                        }

                        // Walk the contiguous data block and collect values
                        let bytes = scope.raw_bytes(&self.interpreter.mem);
                        let mut off = 0usize;
                        while off < bytes.len() {
                            let key_len = bytes[off] as usize;
                            let value_offset = off + 1 + key_len;
                            let val: ShimValue = {
                                let mut val_bytes = [0u8; 8];
                                std::ptr::copy_nonoverlapping(
                                    bytes[value_offset..].as_ptr(),
                                    val_bytes.as_mut_ptr(),
                                    8,
                                );
                                std::mem::transmute(val_bytes)
                            };
                            vals.push(val);
                            off = value_offset + 8;
                        }

                        if scope.parent != 0.into() {
                            vals.push(ShimValue::Environment(scope.parent));
                        }
                    }
                }
            }
        }
    }

    pub fn drop_orphaned_native_types(&mut self) {
        let (to_keep, to_drop): (Vec<_>, Vec<_>) = self.interpreter.mem.droppable_native_pos
            .drain(..)
            .partition(|(pos, _)| self.mask.is_set(*pos as usize));

        self.interpreter.mem.droppable_native_pos = to_keep;
        self.interpreter.mem.dropping_native = true;

        for (pos, type_idx) in to_drop {
            let vtable = self.interpreter.mem.native_type_registry[usize::from(type_idx)].vtable;
            unsafe {
                let data_ptr = self.interpreter.mem.mem().as_ptr().add(pos as usize) as *const ();
                let fat_ptr: (*const (), *const ()) = (data_ptr, vtable);
                let val: &dyn ShimNative = std::mem::transmute(fat_ptr);
                val.gc_drop(self.interpreter);
            }
        }

        self.interpreter.mem.dropping_native = false;
    }

    pub(crate) fn sweep(&mut self) {
        let _zone = zone_scoped!("GC sweep");

        let wilderness = self.interpreter.mem.wilderness;
        let mut new_wilderness = wilderness;
        let mut free_blocks: BTreeMap<u32, Vec<u32>> = BTreeMap::new();

        for range in self.mask.find_zeros() {
            let start = range.start as u32;
            // `find_zeros` rounds the final run up to the next 64-bit boundary,
            // which can extend past the wilderness. Clamp so we never track
            // words that the wilderness allocator also owns (otherwise the same
            // memory could be handed out twice).
            let end = (range.end as u32).min(wilderness);
            if end <= start {
                continue;
            }

            if end == wilderness {
                // The trailing free run is contiguous with the wilderness; fold
                // it back in rather than tracking it as a free block. This keeps
                // the high-water mark accurate and lets the space be reused.
                new_wilderness = start;
                continue;
            }

            free_blocks.entry(end - start).or_default().push(start);
        }

        self.interpreter.mem.free_blocks = free_blocks;
        self.interpreter.mem.wilderness = new_wilderness;
    }
}

pub(crate) struct MemScanner<'a> {
    pub interpreter: &'a mut Interpreter,
    pub mask: Bitmask,
}

    pub(crate) fn scan(
        &mut self,
        mut vals: Vec<ShimValue>
        f: impl Fn<&mut ShimValue, ()>
    ) {
    }
}
