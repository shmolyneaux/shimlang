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
use std::ops::{Deref, DerefMut, Index};
use std::ptr::NonNull;

pub struct ABox<T: ?Sized, A: Allocator>(NonNull<T>, A);

impl<T, A: Allocator> ABox<T, A> {
    pub fn new(val: T, allocator: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<T>();
        let mut ptr = allocator.allocate(layout)?.cast();
        unsafe { *ptr.as_mut() = val };

        Ok(Self(ptr, allocator))
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

pub struct AVecIterator<'a, T, A: Allocator> {
    vec: &'a AVec<T, A>,
    pos: usize
}

impl<'a, T, A: Allocator> AVecIterator<'a, T, A> {
    fn new(vec: &'a AVec<T, A>) -> Self {
        let pos = 0;
        AVecIterator {
            vec,
            pos,
        }
    }
}

impl<'a, T, A: Allocator> Iterator for AVecIterator<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.pos < self.vec.len() {
            self.pos += 1;
            Some(&self.vec[self.pos-1])
        } else {
            None
        }
    }
}

pub struct AVec<T, A: Allocator> {
    ptr: Option<NonNull<T>>,
    len: usize,
    capacity: usize,
    allocator: A,
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

    pub fn iter(&self) -> AVecIterator<'_, T, A> {
        AVecIterator::new(self)
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

        unsafe { *ptr.as_ptr().offset(self.len as isize) = value };

        self.len += 1;

        Ok(())
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
        let mut vec: AVec<u8, _> = AVec::new(allocator);
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
}
