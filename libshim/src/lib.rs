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

pub struct Interpreter<'a, A: Allocator> {
    allocator: A,
    // TODO: figure out how to make the ABox work like this
    print: Option<&'a mut dyn Printer>,
}

pub trait Printer {
    fn print(&mut self, text: &[u8]);
}

impl<'a, A: Allocator> Interpreter<'a, A> {

    pub fn new(allocator: A) -> Interpreter<'a, A> {
        Interpreter { allocator, print: None }
    }

    pub fn set_print_fn(&mut self, f: &'a mut dyn Printer) {
        self.print = Some(f);
    }

    pub fn print(&mut self, text: &[u8]) {
        self.print.as_mut().map(|p| p.print(text));
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
