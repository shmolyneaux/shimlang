#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(dropck_eyepatch)]
#![feature(allocator_api)]
use std::alloc::{AllocError, Allocator, Layout};
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut};
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
