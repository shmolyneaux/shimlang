#![feature(allocator_api)]

use std::alloc::{Allocator, AllocError};
use acollections::ABox;

#[derive(Debug)]
pub enum ShimError {
    AllocError(AllocError)
}

pub enum BinaryOp {
    Add,
    Mul,
}

pub enum Expression<A: Allocator> {
    IntLiteral(i128),
    Binary(BinaryOp, ABox<Expression<A>, A>, ABox<Expression<A>, A>),
}

pub struct Interpreter<A: Allocator> {
    allocator: A,
    // TODO: figure out how to make the ABox work like this
    print_fn: Option<Box<dyn FnMut(&[u8]) -> ()>>,
}

impl<A: Allocator> Interpreter<A> {

    pub fn new(allocator: A) -> Interpreter<A> {
        Interpreter { allocator, print_fn: None }
    }

    pub fn set_print_fn(&mut self, f: Box<dyn FnMut(&[u8]) -> ()>) {
        self.print_fn = Some(f);
    }

    pub fn print(&mut self, text: &[u8]) {
        self.print_fn.as_mut().map(|p| p(text));
    }

    pub fn interpret(&mut self, text: &[u8]) -> Result<(), ShimError> {
        self.print(b"Hello, World!\n");
        Ok(())
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
