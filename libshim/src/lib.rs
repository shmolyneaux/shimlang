#![feature(allocator_api)]

use acollections::{ABox, AVec};
use std::alloc::AllocError;

#[derive(Debug)]
pub enum ShimError {
    PureParseError(PureParseError),
    AllocError(AllocError),
}

impl From<ParseError> for ShimError {
    fn from(err: ParseError) -> ShimError {
        match err {
            ParseError::AllocError(err) => ShimError::AllocError(err),
            ParseError::PureParseError(err) => ShimError::PureParseError(err),
        }
    }
}

// TODO: Forcing the allocator to be copyable doesn't seem right in the general
// case of having sized allocators. It's perfect for zero-sized ones though!
pub trait Allocator: std::alloc::Allocator + Copy {}

#[derive(Debug, Copy, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum Token<'a> {
    Unknown(&'a [u8]),
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
            let chars_left: isize = self.text.len() as isize - self.idx as isize - inc as isize;
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
                        if multichar_ok && self.text[self.idx + inc + 1] == b':' {
                            (Token::DoubleColon, inc + 2)
                        } else {
                            (Token::Colon, inc + 1)
                        }
                    }
                    b'<' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'=' {
                            (Token::Lte, inc + 2)
                        } else {
                            (Token::LeftAngle, inc + 1)
                        }
                    }
                    b'>' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'=' {
                            (Token::Gte, inc + 2)
                        } else {
                            (Token::RightAngle, inc + 1)
                        }
                    }
                    b'-' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'>' {
                            (Token::Arrow, inc + 2)
                        } else {
                            (Token::Minus, inc + 1)
                        }
                    }
                    b'*' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'*' {
                            (Token::DoubleStar, inc + 2)
                        } else {
                            (Token::Star, inc + 1)
                        }
                    }
                    b'=' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'=' {
                            (Token::DoubleEqual, inc + 2)
                        } else {
                            (Token::Equal, inc + 1)
                        }
                    }
                    b'!' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'=' {
                            (Token::BangEqual, inc + 2)
                        } else {
                            (Token::Bang, inc + 1)
                        }
                    }
                    b'/' => {
                        if multichar_ok && self.text[self.idx + inc + 1] == b'/' {
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
                        inc += 1;
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
                        #[derive(Copy, Clone)]
                        enum NumericParseState {
                            LeadingZero,
                            Int,
                            Binary,
                            Hex,
                            Float,
                        }

                        let mut parse_state = if n == 0 {
                            NumericParseState::LeadingZero
                        } else {
                            NumericParseState::Int
                        };

                        inc += 1;

                        let mut float_decimal_power = -1;
                        let mut float_value: f64 = 0.0;
                        let mut int_value: i128 = (n - b'0') as i128;
                        while self.idx + inc < self.text.len() {
                            match (parse_state, self.text[self.idx + inc]) {
                                (NumericParseState::LeadingZero, b'x') => {
                                    parse_state = NumericParseState::Hex;
                                }
                                (NumericParseState::LeadingZero, b'b') => {
                                    parse_state = NumericParseState::Binary;
                                }
                                (NumericParseState::LeadingZero, b'.') => {
                                    float_value = int_value as f64;
                                    parse_state = NumericParseState::Float;
                                }
                                (NumericParseState::LeadingZero, _) => {
                                    break;
                                }
                                (NumericParseState::Int, n) => {
                                    if matches!(n, b'0'..=b'9') {
                                        int_value *= 10;
                                        int_value += (n - b'0') as i128;
                                    } else {
                                        break;
                                    }
                                }
                                (NumericParseState::Binary, n) => {
                                    if matches!(n, b'0' | b'1') {
                                        int_value *= 2;
                                        int_value += (n - b'0') as i128;
                                    } else {
                                        break;
                                    }
                                }
                                (NumericParseState::Hex, n) => {
                                    if matches!(n, b'0'..=b'9') {
                                        int_value *= 16;
                                        int_value += (n - b'0') as i128;
                                    } else if matches!(n, b'a'..=b'f') {
                                        int_value *= 16;
                                        int_value += (n - b'a' + 10) as i128;
                                    } else if matches!(n, b'A'..=b'F') {
                                        int_value *= 16;
                                        int_value += (n - b'A' + 10) as i128;
                                    } else {
                                        break;
                                    }
                                }
                                (NumericParseState::Float, n) => {
                                    // TODO: support 'E' for scientific notation
                                    if matches!(n, b'0'..=b'9') {
                                        float_value += ((n - b'0') as f64)
                                            * (10.0f64).powi(float_decimal_power);
                                        float_decimal_power -= 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            inc += 1;
                        }

                        match parse_state {
                            NumericParseState::Float => (Token::FloatLiteral(float_value), inc),
                            _ => (Token::IntLiteral(int_value), inc),
                        }
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
                    _ => (
                        Token::Unknown(&self.text[self.idx + inc..self.idx + inc + 1]),
                        1,
                    ),
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

pub struct CallExpr<'a, A: Allocator> {
    func: ABox<Expression<'a, A>, A>,
    args: AVec<Expression<'a, A>, A>,
}

pub enum Expression<'a, A: Allocator> {
    Identifier(&'a [u8]),
    IntLiteral(i128),
    Binary(
        BinaryOp,
        ABox<Expression<'a, A>, A>,
        ABox<Expression<'a, A>, A>,
    ),
    Call(CallExpr<'a, A>),
}

#[derive(Debug)]
pub enum PureParseError {
    UnexpectedToken,
    Generic(&'static [u8]),
}

pub enum ParseError {
    AllocError(AllocError),
    PureParseError(PureParseError),
}

impl From<PureParseError> for ParseError {
    fn from(err: PureParseError) -> ParseError {
        ParseError::PureParseError(err)
    }
}

impl From<AllocError> for ParseError {
    fn from(err: AllocError) -> ParseError {
        ParseError::AllocError(err)
    }
}

fn parse_term<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    parse_binary(
        tokens,
        &[(Token::Plus, BinaryOp::Add), (Token::Minus, BinaryOp::Sub)],
        parse_factor,
        allocator,
    )
}

fn parse_factor<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    parse_binary(
        tokens,
        &[(Token::Star, BinaryOp::Mul), (Token::Slash, BinaryOp::Div)],
        parse_call,
        allocator,
    )
}

fn parse_call<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    let mut expr = parse_primary(tokens, allocator)?;
    while tokens.peek() != Token::EOF {
        match tokens.peek() {
            Token::LeftParen => {
                tokens.advance();
                let args = parse_args(tokens, allocator)?;

                if !tokens.matches(Token::RightParen) {
                    return Err(PureParseError::Generic(b"Right paren did not follow arglist").into());
                }

                expr = Expression::Call( CallExpr {
                    func: ABox::new(expr, allocator)?,
                    args: args,
                });
            },
            Token::Dot => {
                // TODO: property access
                return Err(PureParseError::Generic(b"Property access not supported").into());
            },
            _ => {
                break;
            }
        }
    }

    Ok(expr)
}

fn parse_args<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<AVec<Expression<'a, A>, A>, ParseError> {
    let mut args = AVec::new(allocator);
    while tokens.peek() != Token::EOF {
        args.push(parse_expression(tokens, allocator)?);
        if tokens.peek() != Token::Comma {
            break;
        }
    }


    Ok(args)
}

fn parse_primary<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    match tokens.peek() {
        Token::IntLiteral(i) => {
            tokens.advance();
            Ok(Expression::IntLiteral(i))
        }
        Token::Identifier(b"print") => {
            tokens.advance();
            Ok(Expression::Identifier(b"print"))
        },
        other => {
            return Err(PureParseError::Generic(b"Unknown token when parsing primary").into())
        },
    }
}

fn parse_binary<'a, A: Allocator>(
    tokens: &mut TokenStream,
    op_table: &[(Token, BinaryOp)],
    next: fn(&mut TokenStream, A) -> Result<Expression<'a, A>, ParseError>,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    let mut expr = next(tokens, allocator)?;
    while tokens.peek() != Token::EOF {
        let token = tokens.peek();
        if let Some(op) = op_table
            .iter()
            .find(|(table_token, _)| token == *table_token)
            .map(|(_, op)| op)
        {
            // Consume the token we peeked
            tokens.advance();

            let right_expr = next(tokens, allocator)?;
            expr = Expression::Binary(
                *op,
                ABox::new(expr, allocator)?,
                ABox::new(right_expr, allocator)?,
            );
        } else {
            break;
        }
    }

    Ok(expr)
}

fn parse_expression<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<'a, A>, ParseError> {
    parse_term(tokens, allocator)
}

// These values do not get a copy of the Allocator since they are GC'd, and
// the memory is handled externally.
pub enum ShimValue {
    // A variant used to replace a previous-valid value after GC
    Freed,
    // Hard-code this for now until we can declare values
    PrintFn,
    Unit,
    I128(i128)
}

// A newtype which represents an index into a GC'd collection of ShimValue
pub struct Id(usize);

impl Id {
    fn new(id: usize) -> Self {
        Id(id)
    }
}

pub struct Interpreter<'a, A: Allocator> {
    allocator: A,
    values: AVec<ShimValue, A>,
    // TODO: figure out how to make the ABox work like this
    print: Option<&'a mut dyn Printer>,
}

pub trait Printer {
    fn print(&mut self, text: &[u8]);
}

trait NewValue<T> {
    fn new_value(&mut self, val: T) -> Id;
}

impl<'a, A: Allocator> NewValue<i128> for Interpreter<'a, A> {
    fn new_value(&mut self, val: i128) -> Id {
        let id = Id::new(self.values.len());
        self.values.push(ShimValue::I128(val));

        id
    }
}

impl<'a, A: Allocator> NewValue<ShimValue> for Interpreter<'a, A> {
    fn new_value(&mut self, val: ShimValue) -> Id {
        let id = Id::new(self.values.len());
        self.values.push(val);

        id
    }
}

impl<'a, A: Allocator> Interpreter<'a, A> {
    pub fn new(allocator: A) -> Interpreter<'a, A> {
        Interpreter {
            allocator,
            values: AVec::new(allocator),
            print: None,
        }
    }

    pub fn set_print_fn(&mut self, f: &'a mut dyn Printer) {
        self.print = Some(f);
    }

    pub fn print(&mut self, text: &[u8]) {
        self.print.as_mut().map(|p| p.print(text));
    }

    pub fn print_int(&mut self, mut val: i128) {
        if val < 0 {
            self.print(b"-");
            // Hopefully this isn't i128::MIN! (since it can't be expressed as a positive)
            val = val * -1;
        }

        // The largest 128-bit int is 39 characters long
        let mut slice = [b'0'; 39];

        // TODO: check for off-by-one errors in the length
        let mut length = 1;
        for idx in (0..=38).rev() {
            slice[idx] = b'0' + (val % 10) as u8;
            val /= 10;
            if val > 0 {
                length += 1;
            }
        }

        self.print(&slice[39 - length..39]);
    }

    pub fn interpret_expression(&mut self, expr: &Expression<A>) -> Id {
        match expr {
            Expression::Identifier(b"print") => {
                self.new_value(ShimValue::PrintFn)
            }
            Expression::Identifier(_) => {
                self.print(b"Can't interpret identifier\n");
                self.new_value(42)
            }
            Expression::IntLiteral(i) => {
                self.new_value(*i)
            },
            Expression::Binary(op, left, right) => {
                let left = self.interpret_expression(&*left);
                let right = self.interpret_expression(&*right);

                let result = match (&self.values[left.0], &self.values[right.0]) {
                    (ShimValue::I128(a), ShimValue::I128(b)) => {
                        ShimValue::I128(
                            // TODO: wrapping / non-wrapping stuff
                            match op {
                                BinaryOp::Add => a + b,
                                BinaryOp::Sub => a - b,
                                BinaryOp::Mul => a * b,
                                BinaryOp::Div => a / b,
                            }
                        )
                    },
                    _ => {
                        self.print(b"TODO: values can't be added\n");
                        ShimValue::I128(42)
                    }
                };

                self.new_value(result)
            },
            Expression::Call(cexpr) => {
                let func = self.interpret_expression(&cexpr.func);
                match self.values[func.0] {
                    ShimValue::PrintFn => {
                        let last_idx = cexpr.args.len() - 1;
                        for (idx, arg) in cexpr.args.iter().enumerate() {
                            let arg = self.interpret_expression(arg);
                            match self.values[arg.0] {
                                ShimValue::I128(i) => self.print_int(i),
                                _ => self.print(b"other"),
                            }

                            if idx != last_idx {
                                self.print(b" ");
                            }
                        }
                        self.print(b"\n");
                        self.new_value(ShimValue::Unit)
                    },
                    _ => {
                        self.print(b"Can't call value\n");
                        self.new_value(ShimValue::I128(42))
                    }
                }
            }
        }
    }

    pub fn interpret(&mut self, text: &'a [u8]) -> Result<(), ShimError> {
        let mut tokens = TokenStream::new(text);
        let expr = match parse_expression(&mut tokens, self.allocator) {
            Ok(expr) => expr,
            Err(ParseError::PureParseError(PureParseError::Generic(msg))) => {
                self.print(msg);
                return Err(ParseError::PureParseError(PureParseError::Generic(msg)).into());
            }
            Err(err) => {
                self.print(b"Error parsing input");
                return Err(err.into());
            }
        };

        let _ = self.interpret_expression(&expr);

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
