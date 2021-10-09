#![feature(allocator_api)]

use acollections::ABox;
use std::alloc::{AllocError, Allocator};

#[derive(Debug)]
pub enum ShimError {
    AllocError(AllocError),
}

pub enum BinaryOp {
    Add,
    Mul,
}

#[derive(PartialEq, Copy, Clone)]
pub enum Token<'a> {
    Unknown(u8),
    StringLiteral(&'a [u8]),
    UnclosedStringLiteral(&'a [u8]),
    BoolLiteral(bool),
    IntLiteral(i128),
    FloatLiteral(f64),
    Identifier(&'a [u8]),
    LeftParen,
    RightParen,
    LeftCurly,
    RightCurly,
    LeftAngle,
    RightAngle,
    LeftSquare,
    RightSquare,
    Dot,
    Plus,
    Minus,
    Sub,
    Star,
    DoubleStar,
    Slash,
    Colon,
    DoubleColon,
    Semicolon,
    Arrow,
    // TODO: plus_equals, etc.
    DoubleEqual,
    Equal,
    Bang,
    BangEqual,
    Comma,
    Gte,
    Lte,
    AndKeyword,
    AsKeyword,
    BreakKeyword,
    ContinueKeyword,
    ElseKeyword,
    EnumKeyword,
    FnKeyword,
    ForKeyword,
    IfKeyword,
    OrKeyword,
    ReturnKeyword,
    StructKeyword,
    UseKeyword,
    LetKeyword,
    WhileKeyword,
    TrueKeyword,
    FalseKeyword,
    EOF,
}

struct InvalidToken(u8, usize);

struct TokenStream<'a> {
    text: &'a [u8],
    idx: usize,
    cached_next: Option<(Token<'a>, usize)>,
}

impl<'a> TokenStream<'a> {
    pub fn new(text: &'a [u8]) -> Self {
        TokenStream {
            text,
            idx: 0,
            cached_next: None,
        }
    }

    fn peek_inc(&mut self) -> (Token, usize) {
        if let Some(pair) = self.cached_next {
            return pair;
        }

        // Iterate over whitespace until we find a token to return. We accumulate
        // the number of bytes we consume in inc.
        let mut inc = 0;
        loop {
            let chars_left = self.text.len() - self.idx - inc;
            let next = if chars_left <= 0 {
                (Token::EOF, 0)
            } else {
                let multichar_ok = chars_left > 2;
                match self.text[self.idx + inc] {
                    b' ' | b'\t' | b'\r' | b'\n' => {
                        // TODO: line_numbers
                        inc += 1;
                        continue;
                    }
                    b'(' => (Token::LeftParen, inc + 1),
                    b')' => (Token::RightParen, inc + 1),
                    b'{' => (Token::LeftCurly, inc + 1),
                    b'}' => (Token::RightCurly, inc + 1),
                    b'[' => (Token::LeftSquare, inc + 1),
                    b']' => (Token::RightSquare, inc + 1),
                    b'.' => (Token::Dot, inc + 1),
                    b',' => (Token::Comma, inc + 1),
                    b'+' => (Token::Plus, inc + 1),
                    b';' => (Token::Semicolon, inc + 1),
                    b':' => {
                        if multichar_ok && self.text[self.idx + inc] == b':' {
                            (Token::DoubleColon, inc + 2)
                        } else {
                            (Token::Colon, inc + 1)
                        }
                    }
                    b'<' => {
                        if multichar_ok && self.text[self.idx + inc] == b'=' {
                            (Token::Lte, inc + 2)
                        } else {
                            (Token::LeftAngle, inc + 1)
                        }
                    }
                    b'>' => {
                        if multichar_ok && self.text[self.idx + inc] == b'=' {
                            (Token::Gte, inc + 2)
                        } else {
                            (Token::RightAngle, inc + 1)
                        }
                    }
                    b'-' => {
                        if multichar_ok && self.text[self.idx + inc] == b'>' {
                            (Token::Arrow, inc + 2)
                        } else {
                            (Token::Minus, inc + 1)
                        }
                    }
                    b'*' => {
                        if multichar_ok && self.text[self.idx + inc] == b'*' {
                            (Token::DoubleStar, inc + 2)
                        } else {
                            (Token::Star, inc + 1)
                        }
                    }
                    b'=' => {
                        if multichar_ok && self.text[self.idx + inc] == b'=' {
                            (Token::DoubleEqual, inc + 2)
                        } else {
                            (Token::Equal, inc + 1)
                        }
                    }
                    b'!' => {
                        if multichar_ok && self.text[self.idx + inc] == b'=' {
                            (Token::BangEqual, inc + 2)
                        } else {
                            (Token::Bang, inc + 1)
                        }
                    }
                    b'/' => {
                        if multichar_ok && self.text[self.idx + inc] == b'/' {
                            // This is a comment. Consume everything until we reach a newline
                            while self.idx + inc < self.text.len()
                                && self.text[self.idx + inc] != b'\n'
                            {
                                inc += 1;
                            }
                            continue;
                        } else {
                            (Token::Slash, inc + 1)
                        }
                    }
                    b'"' => {
                        while self.idx + inc < self.text.len() && self.text[self.idx + inc] != b'"'
                        {
                            inc += 1;
                        }

                        if self.idx + inc == self.text.len() {
                            // Someone forgot a closing quote :)
                            (
                                Token::UnclosedStringLiteral(&self.text[self.idx + 1..]),
                                inc,
                            )
                        } else {
                            // Consume the closing quote
                            inc += 1;
                            (
                                Token::StringLiteral(&self.text[self.idx + 1..self.idx + inc]),
                                inc,
                            )
                        }
                    }
                    n @ b'0'..=b'9' => {
                        // TODO: Support multi-character numeric literals
                        (Token::IntLiteral((n - b'0').into()), inc + 1)
                    }
                    b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                        while self.idx + inc < self.text.len()
                            && matches!(self.text[self.idx + inc], b'A'..=b'Z' | b'a'..=b'z' | b'_' | b'0'..=b'9' )
                        {
                            inc += 1;
                        }
                        let slice = &self.text[self.idx..self.idx + inc];
                        match slice {
                            b"and" => (Token::AndKeyword, inc),
                            b"as" => (Token::AsKeyword, inc),
                            b"break" => (Token::BreakKeyword, inc),
                            b"continue" => (Token::ContinueKeyword, inc),
                            b"else" => (Token::ElseKeyword, inc),
                            b"enum" => (Token::EnumKeyword, inc),
                            b"fn" => (Token::FnKeyword, inc),
                            b"for" => (Token::ForKeyword, inc),
                            b"in" => (Token::IfKeyword, inc),
                            b"or" => (Token::OrKeyword, inc),
                            b"return" => (Token::ReturnKeyword, inc),
                            b"struct" => (Token::StructKeyword, inc),
                            b"use" => (Token::UseKeyword, inc),
                            b"let" => (Token::LetKeyword, inc),
                            b"while" => (Token::WhileKeyword, inc),
                            b"true" => (Token::BoolLiteral(true), inc),
                            b"false" => (Token::BoolLiteral(false), inc),
                            _ => (Token::Identifier(slice), inc),
                        }
                    }
                    c => (Token::Unknown(c), 1),
                }
            };

            self.cached_next = Some(next);

            return next;
        }
    }

    pub fn peek(&mut self) -> Token {
        self.peek_inc().0
    }

    fn unchecked_advance(&mut self, inc: usize) {
        self.cached_next = None;
        self.idx += inc
    }

    pub fn advance(&mut self) {
        let (_, inc) = self.peek_inc();
        self.unchecked_advance(inc);
    }

    pub fn matches(&mut self, check_token: Token) -> bool {
        let (next_token, inc) = self.peek_inc();
        if std::mem::discriminant(&check_token) == std::mem::discriminant(&next_token) {
            self.unchecked_advance(inc);
            true
        } else {
            false
        }
    }
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
        Interpreter {
            allocator,
            print: None,
        }
    }

    pub fn set_print_fn(&mut self, f: &'a mut dyn Printer) {
        self.print = Some(f);
    }

    pub fn print(&mut self, text: &[u8]) {
        self.print.as_mut().map(|p| p.print(text));
    }

    pub fn interpret(&mut self, text: &[u8]) -> Result<(), ShimError> {
        let mut tokens = TokenStream::new(text);
        let mut token_count = 0;
        while tokens.peek() != Token::EOF {
            tokens.advance();
            token_count += 1;
        }
        self.print(b"Found ");
        self.print(&[token_count + b'0']);
        self.print(b" tokens\n");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn print_int() {
        let text = b"print(3);";
        let mut tokens = TokenStream::new(text);
        assert!(tokens.peek() == Token::Identifier(b"print"));
        tokens.advance();
        assert!(tokens.matches(Token::LeftParen));
        assert!(tokens.peek() == Token::IntLiteral(3));
        tokens.advance();
        assert!(tokens.matches(Token::RightParen));
        assert!(tokens.matches(Token::Semicolon));
        assert!(tokens.matches(Token::EOF));

        // Make sure that `matches` doesn't always return true :)
        assert!(!tokens.matches(Token::DoubleStar));
    }
}
