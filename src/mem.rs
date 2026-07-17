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
    pub wilderness: u32,

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

    pub fn ensure_capacity(&mut self, num_bits: usize) {
        let blocks = num_bits.div_ceil(64);
        while self.data.len() < blocks {
            self.data.push(0);
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

/// Abstraction over the different ways we need to walk every object reachable
/// (transitively) from a set of roots in interpreter memory. There are three
/// callers: the mark phase of the garbage collector and the two hot-reload
/// fix-up passes (struct-shape updates and function-reference updates).
///
/// [`walk_heap`] owns the traversal state — the worklist *and* the mark bitmask
/// (including cycle detection and capacity growth) — and dispatches per
/// [`ShimValue`]. The interpreter being walked is threaded in by `walk_heap`
/// rather than stored here, so implementors only describe what to do with each
/// discovered value; they hold neither the mask nor the interpreter.
pub(crate) trait HeapWalk {
    /// Error produced while visiting a value. The GC never fails
    /// (`Infallible`); hot reload can (`String`).
    type Err;

    /// Resolve the child value stored at word `idx` and return it for
    /// `walk_heap` to enqueue. Hot reload additionally rewrites the slot in
    /// place to point at the reloaded object; the GC just reads it.
    fn visit_slot(&mut self, interp: &mut Interpreter, idx: usize) -> Result<ShimValue, Self::Err>;

    /// Resolve a value living `value_offset` bytes into an env scope's data
    /// block and return it for `walk_heap` to enqueue. The GC just reads it;
    /// hot reload re-homes it, rewrites it, and stores the updated value back
    /// into the scope.
    fn visit_env_value(
        &mut self,
        interp: &mut Interpreter,
        scope_data: u24,
        scope_capacity: u32,
        value_offset: usize,
        val: ShimValue,
    ) -> Result<ShimValue, Self::Err>;
}

/// Mark every word in `range` on `mask`, attaching a debug descriptor built
/// from `desc` only when `gc_debug` is enabled (mirrors [`mark_bit!`]). In
/// release builds `desc` is not evaluated at all, so it may freely format
/// strings or read memory to build a rich description.
macro_rules! mark_range {
    ($mask:expr, $range:expr, $desc:expr) => {{
        #[cfg(feature = "gc_debug")]
        {
            let desc = $desc;
            for idx in $range {
                $mask.set(idx, &desc);
            }
        }
        #[cfg(not(feature = "gc_debug"))]
        {
            for idx in $range {
                $mask.setx(idx);
            }
        }
    }};
}

/// Walk every object transitively reachable from `roots`, dispatching the
/// per-object work to `w`, and return the mark bitmask that records which words
/// were reached. This is the single traversal shared by GC marking and both
/// hot-reload passes; each call gets a fresh mask, so callers never share or
/// manage one themselves. The GC keeps the returned mask (for sweeping and the
/// memory viewer); hot reload only needs its side effects and drops it.
///
/// Safety: reads objects out of interpreter memory using the type tags carried
/// by each [`ShimValue`], so the memory must actually hold values of those
/// shapes (which it does for any live object graph).
pub(crate) fn walk_heap<W: HeapWalk>(
    w: &mut W,
    interp: &mut Interpreter,
    roots: Vec<ShimValue>,
) -> Result<Bitmask, W::Err> {
    let mut mask = Bitmask::new(interp.mem.wilderness as usize);
    // Reserve word 0, the null sentinel reserved by MMU::with_capacity, so the
    // GC never frees it. Harmless for the hot-reload passes that discard `mask`.
    mark_bit!(mask, 0, MemDescriptor::other(0, 1, "null"));

    let mut worklist = roots;
    unsafe {
        while let Some(val) = worklist.pop() {
            // Re-running struct constructors during hot reload can allocate, so
            // keep the mark set large enough for any freshly allocated words.
            mask.ensure_capacity(interp.mem.wilderness as usize);
            match val {
                ShimValue::Integer(_)
                | ShimValue::Float(_)
                | ShimValue::Bool(_)
                | ShimValue::Unit
                | ShimValue::None
                | ShimValue::StopIteration
                | ShimValue::Uninitialized => (),
                ShimValue::Fn(fn_pos) => {
                    let pos: usize = fn_pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let shim_fn_word_count = std::mem::size_of::<ShimFn>().div_ceil(8);
                    let (name_len, name, captured_scope) = {
                        let shim_fn: &ShimFn = interp.mem.get(fn_pos);
                        (shim_fn.name_len, shim_fn.name, shim_fn.captured_scope)
                    };
                    mark_range!(
                        mask,
                        pos..(pos + shim_fn_word_count),
                        MemDescriptor::other(pos, pos + shim_fn_word_count, "ShimFn")
                    );

                    // Mark the function name string
                    worklist.push(ShimValue::String(name_len, 0, name));

                    // Mark the captured scope if present
                    if captured_scope != 0 {
                        worklist.push(ShimValue::Environment(captured_scope.into()));
                    }
                }
                ShimValue::Tuple(len, pos) => {
                    let pos: usize = pos.into();
                    let len: usize = len.into();
                    mark_range!(
                        mask,
                        pos..(pos + len),
                        MemDescriptor::other(pos, pos + len, "Tuple contents")
                    );
                    for idx in pos..(pos + len) {
                        worklist.push(w.visit_slot(interp, idx)?);
                    }
                }
                ShimValue::List(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    mark_range!(
                        mask,
                        pos..(pos + 1),
                        MemDescriptor::other(pos, pos + 1, "List header")
                    );

                    let (data, len, capacity) = {
                        let lst: &ShimList = interp.mem.get(pos.into());
                        (usize::from(lst.data), lst.len(), lst.capacity())
                    };

                    mark_range!(
                        mask,
                        data..(data + capacity),
                        MemDescriptor::other(data, data + capacity, "List item")
                    );

                    // Only the logically-present elements are live; the unused
                    // tail of the backing store is marked (above) but not walked.
                    for idx in data..(data + len) {
                        worklist.push(w.visit_slot(interp, idx)?);
                    }
                }
                s @ ShimValue::String(len, offset, pos) => {
                    let pos: usize = usize::from(pos);
                    if mask.is_set(pos) {
                        continue;
                    }
                    let len = len as usize;
                    let offset = offset as usize;
                    #[cfg(not(feature = "gc_debug"))]
                    let _ = s;
                    mark_range!(
                        mask,
                        pos..(pos + (offset + len).div_ceil(8)),
                        MemDescriptor::other(
                            pos,
                            (offset + len).div_ceil(8),
                            &format!(
                                "String: {}",
                                debug_u8s(s.string_from_mem(&interp.mem).unwrap())
                            )
                        )
                    );
                }
                ShimValue::Dict(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }

                    let (entries_base, entry_count) = {
                        let dict: &ShimDict = interp.mem.get(pos.into());
                        (usize::from(dict.entries), dict.entry_count as usize)
                    };

                    // Collect the keys and the word index of each value slot,
                    // then release the borrow before touching the walker again.
                    // `DictEntry` has no `repr(C)`, so ask the compiler where the
                    // value field actually lives rather than assuming an order.
                    let entry_words = std::mem::size_of::<DictEntry>() / 8;
                    let value_word = std::mem::offset_of!(DictEntry, value) / 8;
                    let mut keys: Vec<ShimValue> = Vec::new();
                    let mut value_slots: Vec<usize> = Vec::new();
                    {
                        let u64_slice = &interp.mem.mem()
                            [entries_base..entries_base + entry_words * entry_count];
                        let entries: &[DictEntry] = std::slice::from_raw_parts(
                            u64_slice.as_ptr() as *const DictEntry,
                            u64_slice.len() / entry_words,
                        );
                        for (entry_idx, entry) in entries[..entry_count].iter().enumerate() {
                            if !entry.key.is_uninitialized() {
                                keys.push(entry.key);
                                value_slots
                                    .push(entries_base + entry_idx * entry_words + value_word);
                            }
                        }
                    }
                    // Keys are never structs or functions (not hashable), so
                    // they only need to be traversed, not rewritten.
                    for key in keys {
                        worklist.push(key);
                    }
                    for slot in value_slots {
                        worklist.push(w.visit_slot(interp, slot)?);
                    }

                    // Mark the dict header.
                    mark_range!(
                        mask,
                        pos..(pos + std::mem::size_of::<ShimDict>().div_ceil(8)),
                        MemDescriptor::other(
                            pos,
                            pos + std::mem::size_of::<ShimDict>().div_ceil(8),
                            "dict header"
                        )
                    );

                    let (indices_pos, indices_word_count, entries_pos, entries_span) = {
                        let dict: &ShimDict = interp.mem.get(pos.into());
                        let size: usize = 1 << dict.size_pow;
                        let indices_word_count = if dict.size_pow == 0 {
                            0
                        } else {
                            size.div_ceil(8 / dict.indices_stride_bytes(size))
                        };
                        (
                            usize::from(dict.indices),
                            indices_word_count,
                            usize::from(dict.entries),
                            dict.capacity() * 3,
                        )
                    };

                    // Mark the indices array
                    mark_range!(
                        mask,
                        indices_pos..(indices_pos + indices_word_count),
                        MemDescriptor::other(
                            indices_pos,
                            indices_pos + indices_word_count,
                            "dict index"
                        )
                    );

                    // Mark the entries array
                    mark_range!(
                        mask,
                        entries_pos..(entries_pos + entries_span),
                        MemDescriptor::other(entries_pos, entries_pos + entries_span, "dict entries")
                    );
                }
                ShimValue::Set(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let dict_pos = {
                        let set: &ShimSet = interp.mem.get(pos.into());
                        set.dict_pos
                    };

                    // Mark the space for the ShimSet struct
                    mark_range!(
                        mask,
                        pos..(pos + std::mem::size_of::<ShimSet>().div_ceil(8)),
                        MemDescriptor::other(
                            pos,
                            pos + std::mem::size_of::<ShimSet>().div_ceil(8),
                            "set header"
                        )
                    );

                    // And walk the dict as normal. An empty set has no backing
                    // dict allocated yet (dict_pos == 0), so nothing more to do.
                    if dict_pos != u24::from(0) {
                        worklist.push(ShimValue::Dict(dict_pos));
                    }
                }
                ShimValue::StructDef(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let (mem_size, method_positions): (usize, Vec<u24>) = {
                        let def: &StructDef = interp.mem.get(pos.into());
                        (def.mem_size(), def.method_fn_positions().collect())
                    };
                    mark_range!(
                        mask,
                        pos..(pos + mem_size),
                        {
                            let def: &StructDef = interp.mem.get(pos.into());
                            MemDescriptor::other(
                                pos,
                                pos + mem_size,
                                &format!("struct {}", debug_u8s(&def.name)),
                            )
                        }
                    );

                    // Mark method functions referenced by this struct def
                    for fn_pos in method_positions {
                        worklist.push(ShimValue::Fn(fn_pos));
                    }
                }
                ShimValue::Struct(def_pos, pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let member_count = {
                        let def: &StructDef = interp.mem.get(def_pos);
                        def.member_count as usize
                    };
                    mark_range!(
                        mask,
                        pos..(pos + member_count),
                        {
                            let interp = &*interp;
                            let def: &StructDef = interp.mem.get(def_pos);
                            let mut members = Vec::new();
                            for idx in pos..(pos + member_count) {
                                members.push(ShimValue::from_u64(interp.mem.mem()[idx]));
                            }
                            MemDescriptor::struct_desc(
                                pos,
                                pos + member_count,
                                format!("struct {}", debug_u8s(&def.name)),
                                members,
                            )
                        }
                    );
                    for idx in pos..(pos + member_count) {
                        worklist.push(w.visit_slot(interp, idx)?);
                    }
                    worklist.push(ShimValue::StructDef(def_pos));
                }
                ShimValue::NativeFn(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    mark_range!(
                        mask,
                        pos..(pos + 1),
                        MemDescriptor::other(pos, pos + 1, "NativeFn")
                    );
                }
                ShimValue::Native(type_idx, pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let type_idx: usize = type_idx.into();
                    let (native_word_count, vtable) = {
                        let info = &interp.mem.native_type_registry[type_idx];
                        (info.word_count, info.vtable)
                    };
                    mark_range!(
                        mask,
                        pos..(pos + native_word_count),
                        MemDescriptor::other(pos, pos + native_word_count, "Native")
                    );

                    // Reconstruct a fat pointer to call gc_vals() without a Box.
                    let extra = {
                        let data_ptr =
                            &interp.mem.mem()[pos] as *const u64 as *const ();
                        let fat_ptr: (*const (), *const ()) = (data_ptr, vtable);
                        let native_ref: &dyn ShimNative = std::mem::transmute(fat_ptr);
                        native_ref.gc_vals()
                    };
                    worklist.extend(extra);
                }
                ShimValue::BoundMethod(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    // Mark the 2 words used to store the BoundMethod
                    mark_range!(
                        mask,
                        pos..(pos + 1),
                        MemDescriptor::other(pos, pos + 1, "Bound Method Obj")
                    );
                    mark_range!(
                        mask,
                        (pos + 1)..(pos + 2),
                        MemDescriptor::other(pos + 1, pos + 2, "Bound Method fn")
                    );

                    // word[pos] holds the bound object; word[pos + 1] holds the fn.
                    worklist.push(w.visit_slot(interp, pos)?);
                    let func_pos = u24::from(interp.mem.mem()[pos + 1]);
                    worklist.push(ShimValue::Fn(func_pos));
                }
                ShimValue::BoundNativeMethod(pos) => {
                    let pos: usize = pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    // Two words: the bound ShimValue and the native fn pointer.
                    mark_range!(
                        mask,
                        pos..(pos + 2),
                        MemDescriptor::other(pos, pos + 2, "BoundNativeMethod")
                    );

                    // word[pos] is the bound object; walk/rewrite it.
                    worklist.push(w.visit_slot(interp, pos)?);
                }
                ShimValue::Environment(scope_pos) => {
                    let pos: usize = scope_pos.into();
                    if mask.is_set(pos) {
                        continue;
                    }
                    let (scope_data, scope_capacity, scope_used, scope_parent) = {
                        let scope: &EnvScope = interp.mem.get(scope_pos);
                        (scope.data, scope.capacity, scope.used(), scope.parent)
                    };

                    // Chunk of memory that stores the EnvScope metadata
                    mark_range!(
                        mask,
                        pos..(pos + std::mem::size_of::<EnvScope>().div_ceil(8)),
                        MemDescriptor::env_header(
                            pos,
                            pos + std::mem::size_of::<EnvScope>().div_ceil(8),
                            &format!("Envscope header\nparent: {:?}", scope_parent)
                        )
                    );

                    // Data block
                    let start = usize::from(scope_data);
                    let end = start + scope_capacity as usize;
                    mark_range!(
                        mask,
                        start..end,
                        {
                            let interp = &*interp;
                            let scope: &EnvScope = interp.mem.get(scope_pos);
                            MemDescriptor::env_data(start, end, &scope.to_string(&interp.mem))
                        }
                    );

                    // Walk the contiguous data block and hand each value to the
                    // walker. Each read re-derives a short-lived byte view
                    // instead of holding one across the loop, since hot reload's
                    // visit_env_value needs a mutable borrow of memory.
                    let mut off = 0usize;
                    while off < scope_used as usize {
                        let (value_offset, val) = {
                            let word_count = (scope_used as usize).div_ceil(8);
                            let u64_slice = &interp.mem.mem()[start..start + word_count];
                            let bytes: &[u8] = std::slice::from_raw_parts(
                                u64_slice.as_ptr() as *const u8,
                                scope_used as usize,
                            );
                            let key_len = bytes[off] as usize;
                            let value_offset = off + 1 + key_len;
                            let mut val_bytes = [0u8; 8];
                            std::ptr::copy_nonoverlapping(
                                bytes[value_offset..].as_ptr(),
                                val_bytes.as_mut_ptr(),
                                8,
                            );
                            let val: ShimValue = std::mem::transmute(val_bytes);
                            (value_offset, val)
                        };

                        let enqueued = w.visit_env_value(
                            interp,
                            scope_data,
                            scope_capacity,
                            value_offset,
                            val,
                        )?;
                        worklist.push(enqueued);

                        off = value_offset + 8;
                    }

                    if scope_parent != 0.into() {
                        worklist.push(ShimValue::Environment(scope_parent));
                    }
                }
            }
        }
    }

    Ok(mask)
}

pub(crate) struct GC<'a> {
    pub interpreter: &'a mut Interpreter,
}

impl<'a> GC<'a> {
    pub(crate) fn new(interpreter: &'a mut Interpreter) -> Self {
        Self { interpreter }
    }

    /// Mark every object reachable from `roots`, returning the resulting mark
    /// bitmask for [`GC::sweep`] / [`GC::drop_orphaned_native_types`].
    pub(crate) fn mark(&mut self, roots: Vec<ShimValue>) -> Bitmask {
        let _zone = zone_scoped!("GC mark");
        // The GC never fails while marking; its `HeapWalk::Err` is `Infallible`.
        walk_heap(&mut GcWalk, &mut *self.interpreter, roots).unwrap()
    }

    pub fn drop_orphaned_native_types(&mut self, mask: &Bitmask) {
        let (to_keep, to_drop): (Vec<_>, Vec<_>) = self.interpreter.mem.droppable_native_pos
            .drain(..)
            .partition(|(pos, _)| mask.is_set(*pos as usize));

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

    pub(crate) fn sweep(&mut self, mask: &Bitmask) {
        let _zone = zone_scoped!("GC sweep");

        let wilderness = self.interpreter.mem.wilderness;
        let mut new_wilderness = wilderness;
        let mut free_blocks: BTreeMap<u32, Vec<u32>> = BTreeMap::new();

        for range in mask.find_zeros() {
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

/// The [`HeapWalk`] the garbage collector uses to mark. It holds no state — the
/// interpreter is threaded in by [`walk_heap`] — so it stays separate from
/// [`GC`], which owns the interpreter for the sweep/drop phases.
struct GcWalk;

impl HeapWalk for GcWalk {
    // Marking is infallible.
    type Err = std::convert::Infallible;

    fn visit_slot(&mut self, interp: &mut Interpreter, idx: usize) -> Result<ShimValue, Self::Err> {
        Ok(unsafe { *interp.mem.get(idx.into()) })
    }

    fn visit_env_value(
        &mut self,
        _interp: &mut Interpreter,
        _scope_data: u24,
        _scope_capacity: u32,
        _value_offset: usize,
        val: ShimValue,
    ) -> Result<ShimValue, Self::Err> {
        Ok(val)
    }
}
