#![feature(dropck_eyepatch)]
#![feature(allocator_api)]

/// You should probably not use this unless you're me. Even if you are me, it's
/// probably a bad idea then too. The standard library exists for a reason, it
/// just hasn't quite figured out fallible allocations yet.
///
/// The invariants that unsafe Rust need to uphold are numerous. Things that
/// seem perfectly reasonable in C wouldn't fly in Rust because of the safety
/// guarantees that are _required_ from unsafe code.
///
/// On top of that, the standard library uses a bunch of special internal
/// features/types to make writing memory abstractions easier, and not all of
/// those are exposed (looking at you RawVec 👀)
use std::alloc::{AllocError, Allocator, Layout};
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

#[derive(Debug)]
pub struct ABox<T: ?Sized, A: Allocator>(NonNull<T>, A);

pub trait AClone {
    fn aclone(&self) -> Result<Self, AllocError>
    where
        Self: Sized;
}

impl<T: Clone + Sized> AClone for T {
    fn aclone(&self) -> Result<Self, AllocError> {
        Ok(self.clone())
    }
}

impl<T, A: Allocator> ABox<T, A> {
    pub fn new(val: T, allocator: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = allocator.allocate(layout)?.cast();
        unsafe { ptr.as_ptr().write(val) };

        Ok(Self(ptr, allocator))
    }

    pub fn into_inner(self) -> T {
        // Read the value, and have the pointer get freed
        let val = unsafe { self.0.as_ptr().read() };

        // TODO: we don't actually read layout in the allocator, so we're giving
        // some garbage information since we can't get the layout for unsized
        // types, and I don't otherwise know how to deallocate.
        let layout = Layout::new::<u8>();
        unsafe { self.1.deallocate(self.0.cast(), layout) };

        // We don't want to drop self since we already deallocated the pointer.
        std::mem::forget(self);

        val
    }
}

unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for ABox<T, A> {
    fn drop(&mut self) {
        // This makes drops sound complicated, and I don't understand what I
        // actually need to understand:
        // https://forge.rust-lang.org/libs/maintaining-std.html#is-there-a-manual-drop-implementation

        unsafe { std::ptr::drop_in_place(self.0.as_ptr()) };

        // TODO: we don't actually read layout in the allocator, so we're giving
        // some garbage information since we can't get the layout for unsized
        // types, and I don't otherwise know how to deallocate.
        let layout = Layout::new::<u8>();
        unsafe { self.1.deallocate(self.0.cast(), layout) };
    }
}

impl<T: ?Sized, A: Allocator> Deref for ABox<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.0.as_ptr() }
    }
}

impl<T: ?Sized, A: Allocator> DerefMut for ABox<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0.as_mut() }
    }
}

impl<T: ?Sized, A: Allocator> AsMut<T> for ABox<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized, A: Allocator> AsRef<T> for ABox<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Allocator> Borrow<T> for ABox<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Allocator> BorrowMut<T> for ABox<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: AClone, A: Clone + Allocator> AClone for ABox<T, A> {
    fn aclone(&self) -> Result<Self, AllocError> {
        let val: T = self.deref().aclone()?;
        let new_box: ABox<T, A> = ABox::new(val, self.1.clone())?;

        Ok(new_box)
    }
}

#[derive(Debug)]
pub struct AVec<T, A: Allocator> {
    ptr: Option<NonNull<T>>,
    len: usize,
    capacity: usize,
    pub allocator: A,
}

impl<T, A: Allocator> AVec<T, A> {
    pub fn new(allocator: A) -> Self {
        AVec {
            allocator,
            ptr: None,
            len: 0,
            capacity: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        let slice: &[T] = self;
        slice.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        let slice: &mut [T] = self;
        slice.iter_mut()
    }

    pub fn push(&mut self, value: T) -> Result<(), AllocError> {
        if self.len == self.capacity {
            let additional = if self.capacity == 0 {
                // Start with 4 elements (arbitrary)
                4
            } else {
                // Otherwise double the capacity
                self.capacity
            };
            self.reserve(additional)?;
        }

        // We either had capacity already or we reserved some without error
        assert!(self.len < self.capacity);

        // Since we have capacity, we must have a valid pointer
        let ptr = self.ptr.unwrap();

        unsafe { ptr.as_ptr().offset(self.len as isize).write(value) };

        self.len += 1;

        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);

        unsafe {
            // Since we have an index less than len, we must have a valid pointer
            let ptr = self.ptr.unwrap().as_ptr().offset(index as isize);
            let value = std::ptr::read(ptr);

            std::ptr::copy(ptr.offset(1), ptr, self.len - index - 1);

            self.len -= 1;

            value
        }
    }

    pub fn reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        let bytes = if let Some(ptr) = self.ptr {
            // We ignore LayoutError's under the assumption that they're very unlikely
            let old_layout = Layout::array::<T>(self.capacity).unwrap();
            let new_layout = Layout::array::<T>(self.capacity + additional).unwrap();
            unsafe { self.allocator.grow(ptr.cast(), old_layout, new_layout)? }
        } else {
            // We ignore LayoutError's under the assumption that they're very unlikely
            let layout = Layout::array::<T>(additional).unwrap();
            self.allocator.allocate(layout)?
        };

        // TODO: actually use the number of bytes we got back, which may
        // be more than we asked for.
        self.ptr = Some(bytes.cast());
        self.capacity += additional;

        Ok(())
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len {
            None
        } else {
            Some(&self[idx])
        }
    }
}

impl<T: AClone, A: Clone + Allocator> AClone for AVec<T, A> {
    fn aclone(&self) -> Result<Self, AllocError> {
        let mut new_vec = AVec::new(self.allocator.clone());
        new_vec.extend_from_slice(self)?;

        Ok(new_vec)
    }
}

impl<T: AClone, A: Allocator> AVec<T, A> {
    pub fn from_slice(slice: &[T], allocator: A) -> Result<Self, AllocError> {
        let mut new_vec = Self::new(allocator);
        new_vec.extend_from_slice(slice)?;

        Ok(new_vec)
    }

    pub fn extend_from_slice(&mut self, other: &[T]) -> Result<(), AllocError> {
        for item in other {
            self.push(item.aclone()?)?;
        }
        Ok(())
    }
}

impl<T, A: Allocator> Deref for AVec<T, A> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        if let Some(ptr) = self.ptr {
            unsafe { &*std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), self.len) }
        } else {
            &[]
        }
    }
}

impl<T, A: Allocator> DerefMut for AVec<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        if let Some(ptr) = self.ptr {
            unsafe { &mut *std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), self.len) }
        } else {
            &mut []
        }
    }
}

unsafe impl<#[may_dangle] T, A: Allocator> Drop for AVec<T, A> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            unsafe {
                std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), self.len))
            };

            // We ignore LayoutError's under the assumption that they're very unlikely
            let layout = Layout::array::<T>(self.capacity).unwrap();
            unsafe { self.allocator.deallocate(ptr.cast(), layout) };
        }
    }
}

impl<T, A: Allocator> Index<usize> for AVec<T, A> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        assert!(idx < self.len);

        // Since we have length, we must have a valid pointer
        let ptr = self.ptr.unwrap();

        unsafe { &*ptr.as_ptr().offset(idx as isize) }
    }
}

impl<T, A: Allocator> IndexMut<usize> for AVec<T, A> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len);

        // Since we have length, we must have a valid pointer
        let ptr = self.ptr.unwrap();

        unsafe { &mut *ptr.as_ptr().offset(idx as isize) }
    }
}

impl<A: Allocator> std::io::Write for AVec<u8, A> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        self.extend_from_slice(buf)
            .map_err(|alloc_err| std::io::Error::new(std::io::ErrorKind::OutOfMemory, alloc_err))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

impl<T: std::cmp::PartialEq, A: Allocator> std::cmp::PartialEq for AVec<T, A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len == other.len {
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
        } else {
            false
        }
    }
}

impl<T, A: Allocator> Borrow<[T]> for AVec<T, A> {
    fn borrow(&self) -> &[T] {
        self.deref()
    }
}

#[derive(Debug)]
pub struct AHashEntry<K, V> {
    key: K,
    value: V,
}

impl<K: std::cmp::PartialEq, V> AHashEntry<K, V> {
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn value(&self) -> &V {
        &self.value
    }
}

// TODO: Actually make this a hashmap, rather than an associative array
#[derive(Debug)]
pub struct AHashMap<K: std::cmp::PartialEq, V, A: Allocator> {
    vec: AVec<AHashEntry<K, V>, A>,
}

impl<K: std::cmp::PartialEq, V, A: Allocator> AHashMap<K, V, A> {
    pub fn new(allocator: A) -> Self {
        let vec = AVec::new(allocator);
        AHashMap { vec }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: std::cmp::PartialEq + ?Sized,
    {
        self.get_entry(key).map(|entry| &entry.value)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: std::cmp::PartialEq + ?Sized,
    {
        self.get_entry_mut(key).map(|entry| &mut entry.value)
    }

    pub fn get_entry<Q>(&self, key: &Q) -> Option<&AHashEntry<K, V>>
    where
        K: Borrow<Q>,
        Q: std::cmp::PartialEq + ?Sized,
    {
        self.vec.iter().find(|entry| entry.key.borrow() == key)
    }

    pub fn get_entry_mut<Q>(&mut self, key: &Q) -> Option<&mut AHashEntry<K, V>>
    where
        K: Borrow<Q>,
        Q: std::cmp::PartialEq + ?Sized,
    {
        self.vec.iter_mut().find(|entry| entry.key.borrow() == key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, AllocError> {
        if let Some(entry) = self.get_entry_mut(&key) {
            Ok(Some(std::mem::replace(&mut entry.value, value)))
        } else {
            self.vec.push(AHashEntry { key, value })?;
            Ok(None)
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(idx) = self
            .vec
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.key.borrow() == key)
            .map(|(idx, _)| idx)
        {
            Some(self.vec.remove(idx).value)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &AHashEntry<K, V>> + '_ {
        self.vec.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn vec_basic() {
        let allocator = std::alloc::Global;
        let mut vec: AVec<u8, _> = AVec::new(allocator);
        vec.push(3).unwrap();
        vec.push(2).unwrap();
        vec.push(1).unwrap();

        assert_eq!(vec[0], 3);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 1);
    }

    #[test]
    fn vec_alloc_many() {
        let allocator = std::alloc::Global;
        let mut vec: AVec<u32, _> = AVec::new(allocator);

        // Presumebly this will need to reallocate many times
        for i in 0..10000 {
            vec.push(i).unwrap();
        }

        assert_eq!(vec[5555], 5555);
    }

    #[test]
    fn vec_iter() {
        let allocator = std::alloc::Global;
        let mut vec: AVec<u8, _> = AVec::new(allocator);
        vec.push(3).unwrap();
        vec.push(2).unwrap();
        vec.push(1).unwrap();

        let mut iter = vec.iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic]
    fn vec_panic_when_empty() {
        let allocator = std::alloc::Global;
        let vec: AVec<u8, _> = AVec::new(allocator);
        vec[0];
    }

    #[test]
    #[should_panic]
    fn vec_panic_after_push() {
        let allocator = std::alloc::Global;
        let mut vec: AVec<u8, _> = AVec::new(allocator);
        vec.push(0).unwrap();

        vec[1];
    }

    #[test]
    fn hashmap_basic() {
        let allocator = std::alloc::Global;
        let mut map = AHashMap::new(allocator);
        map.insert(1, "world".to_string()).unwrap();
        map.insert(0, "hello".to_string()).unwrap();

        assert_eq!(map.get(&0), Some(&"hello".to_string()));
        assert_eq!(map.get(&1), Some(&"world".to_string()));
        assert_eq!(map.get(&-1), None);
        assert_eq!(map.remove(&1), Some("world".to_string()));
        assert_eq!(map.remove(&1), None);
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn hashmap_avec_key() {
        let allocator = std::alloc::Global;
        let mut map = AHashMap::new(allocator);
        let mut one = AVec::new(allocator);
        one.extend_from_slice(b"one").unwrap();
        map.insert(one, 1).unwrap();

        let key: &[u8] = b"one";
        assert_eq!(map.get(key), Some(&1));
    }
}
