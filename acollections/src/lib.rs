#![feature(allocator_api)]
use std::alloc::{AllocError, Allocator, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub struct ABox<T, A: Allocator>(NonNull<T>, A);

impl<T, A: Allocator> ABox<T, A> {
    pub fn new(val: T, allocator: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<T>();
        let mut ptr = allocator.allocate(layout)?.cast();
        unsafe { *ptr.as_mut() = val };

        Ok(Self(ptr, allocator))
    }
}

impl<T, A: Allocator> Drop for ABox<T, A> {
    fn drop(&mut self) {
        let layout = Layout::new::<T>();
        unsafe { self.1.deallocate(self.0.cast(), layout) };
    }
}

impl<T, A: Allocator> Deref for ABox<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.0.as_ptr() }
    }
}

impl<T, A: Allocator> DerefMut for ABox<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0.as_mut() }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
