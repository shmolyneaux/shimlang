#![feature(allocator_api)]

use acollections::{ABox, AVec, AHashMap};
use std::alloc::AllocError;

use lexical_core::FormattedSize;

#[derive(Debug)]
pub enum ShimError {
    PureParseError(PureParseError),
    AllocError(AllocError),
    Other(&'static [u8]),
}

impl From<ParseError> for ShimError {
    fn from(err: ParseError) -> ShimError {
        match err {
            ParseError::AllocError(err) => ShimError::AllocError(err),
            ParseError::PureParseError(err) => ShimError::PureParseError(err),
        }
    }
}

impl From<AllocError> for ShimError {
    fn from(err: AllocError) -> ShimError {
        ShimError::AllocError(err)
    }
}

// TODO: Forcing the allocator to be copyable doesn't seem right in the general
// case of having sized allocators. It's perfect for zero-sized ones though!
pub trait Allocator: std::alloc::Allocator + Copy + std::fmt::Debug {}

#[derive(Debug, Copy, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
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
                        let start_idx = self.idx + inc;
                        // Keep going if:
                        // - There's more text, and
                        //   - It's not a quote, or
                        //   - The quote is escaped
                        // NOTE: the escaped quote will be handled by the parser
                        while self.idx + inc < self.text.len()
                            && (self.text[self.idx + inc - 1] == b'\\'
                                || self.text[self.idx + inc] != b'"')
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
                                // Subtract 1 from `inc` to _not_ include the closing quote
                                Token::StringLiteral(&self.text[start_idx..self.idx + inc - 1]),
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
                                (NumericParseState::LeadingZero | NumericParseState::Int, b'.') => {
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
                        let start_idx = self.idx + inc;

                        while self.idx + inc < self.text.len()
                            && matches!(self.text[self.idx + inc], b'A'..=b'Z' | b'a'..=b'z' | b'_' | b'0'..=b'9' )
                        {
                            inc += 1;
                        }
                        let slice = &self.text[start_idx..self.idx + inc];
                        match slice {
                            b"and" => (Token::AndKeyword, inc),
                            b"as" => (Token::AsKeyword, inc),
                            b"break" => (Token::BreakKeyword, inc),
                            b"continue" => (Token::ContinueKeyword, inc),
                            b"else" => (Token::ElseKeyword, inc),
                            b"enum" => (Token::EnumKeyword, inc),
                            b"fn" => (Token::FnKeyword, inc),
                            b"for" => (Token::ForKeyword, inc),
                            b"if" => (Token::IfKeyword, inc),
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

#[derive(Debug)]
pub struct CallExpr<A: Allocator> {
    func: ABox<Expression<A>, A>,
    args: AVec<Expression<A>, A>,
}

#[derive(Debug)]
pub enum Expression<A: Allocator> {
    Identifier(AVec<u8, A>),
    BoolLiteral(bool),
    IntLiteral(i128),
    FloatLiteral(f64),
    StringLiteral(AVec<u8, A>),
    Binary(
        BinaryOp,
        ABox<Expression<A>, A>,
        ABox<Expression<A>, A>,
    ),
    Call(CallExpr<A>),
}

#[derive(Debug)]
pub struct Block<A: Allocator> {
    stmts: AVec<Statement<A>, A>,
}

#[derive(Debug)]
pub struct IfStatement<A: Allocator> {
    predicate: Expression<A>,
    if_block: Block<A>,
    else_block: Option<Block<A>>,
}

#[derive(Debug)]
pub enum Statement<A: Allocator> {
    Expression(Expression<A>),
    Declaration(AVec<u8, A>, Expression<A>),
    Assignment(AVec<u8, A>, Expression<A>),
    Block(Block<A>),
    IfStatement(IfStatement<A>),
    WhileStatement(Expression<A>, Block<A>),
    BreakStatement,
    ContinueStatement,
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

fn parse_equality<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
    parse_binary(
        tokens,
        &[(Token::DoubleEqual, BinaryOp::Eq), (Token::BangEqual, BinaryOp::Neq)],
        parse_comparison,
        allocator,
    )
}

fn parse_comparison<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
    parse_binary(
        tokens,
        &[
            (Token::LeftAngle, BinaryOp::Lt),
            (Token::Lte, BinaryOp::Lte),
            (Token::RightAngle, BinaryOp::Gt),
            (Token::Gte, BinaryOp::Gte),
        ],
        parse_term,
        allocator,
    )
}

fn parse_term<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
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
) -> Result<Expression<A>, ParseError> {
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
) -> Result<Expression<A>, ParseError> {
    let mut expr = parse_primary(tokens, allocator)?;
    while tokens.peek() != Token::EOF {
        match tokens.peek() {
            Token::LeftParen => {
                tokens.advance();
                let args = parse_args(tokens, allocator)?;

                if !tokens.matches(Token::RightParen) {
                    return Err(
                        PureParseError::Generic(b"Right paren did not follow arglist").into(),
                    );
                }

                expr = Expression::Call(CallExpr {
                    func: ABox::new(expr, allocator)?,
                    args: args,
                });
            }
            Token::Dot => {
                // TODO: property access
                return Err(PureParseError::Generic(b"Property access not supported").into());
            }
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
) -> Result<AVec<Expression<A>, A>, ParseError> {
    let mut args = AVec::new(allocator);
    while tokens.peek() != Token::EOF && tokens.peek() != Token::RightParen {
        args.push(parse_expression(tokens, allocator)?)?;
        if tokens.matches(Token::Comma) {
            continue;
        }
        break;
    }

    Ok(args)
}

fn parse_primary<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
    match tokens.peek() {
        Token::BoolLiteral(b) => {
            tokens.advance();
            Ok(Expression::BoolLiteral(b))
        }
        Token::IntLiteral(i) => {
            tokens.advance();
            Ok(Expression::IntLiteral(i))
        }
        Token::FloatLiteral(f) => {
            tokens.advance();
            Ok(Expression::FloatLiteral(f))
        }
        Token::StringLiteral(s) => {
            let mut new_str = AVec::new(allocator);
            let mut slash = false;
            for c in s {
                new_str.push(match (c, slash) {
                    (b'\\', false) => {
                        slash = true;
                        continue;
                    }
                    (b'n', true) => b'\n',
                    (b't', true) => b'\t',
                    (b'r', true) => b'\r',
                    _ => *c,
                })?;
                slash = false;
            }

            tokens.advance();
            Ok(Expression::StringLiteral(new_str))
        }
        Token::Identifier(ident) => {
            let mut vec = AVec::new(allocator);
            vec.extend_from_slice(ident)?;

            tokens.advance();
            Ok(Expression::Identifier(vec))
        }
        token => {
            if cfg!(debug_assertions) {
                println!("Token: {:?}", token);
            }

            Err(PureParseError::Generic(b"Unknown token when parsing primary").into())
        },
    }
}

fn parse_binary<'a, A: Allocator>(
    tokens: &mut TokenStream,
    op_table: &[(Token, BinaryOp)],
    next: fn(&mut TokenStream, A) -> Result<Expression<A>, ParseError>,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
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

fn parse_script<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Block<A>, ParseError> {
    // TODO: shebang
    let block = parse_block_open(tokens, allocator)?;
    if tokens.matches(Token::EOF) {
        Ok(block)
    } else {
        Err(PureParseError::Generic(b"Unconsumed tokens!").into())
    }
}

// NOTE: consumes open and closing curly
fn parse_block<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Block<A>, ParseError> {
    if !tokens.matches(Token::LeftCurly) {
        if cfg!(debug_assertions) {
            let token = tokens.peek();
            println!("Token: {:?}", token);
        }
        return Err(PureParseError::Generic(b"Block does not start with opening '{'").into());
    }

    let block = parse_block_open(tokens, allocator)?;

    if !tokens.matches(Token::RightCurly) {
        return Err(PureParseError::Generic(b"Block does not end with closing '}'").into());
    }

    Ok(block)
}

fn parse_block_open<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Block<A>, ParseError> {
    let mut stmts = AVec::new(allocator);
    while tokens.peek() != Token::EOF && tokens.peek() != Token::RightCurly {
        let stmt = parse_statement(tokens, allocator)?;
        stmts.push(stmt)?;
    }

    Ok(Block { stmts })
}

fn parse_if<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Statement<A>, ParseError> {
    if tokens.matches(Token::IfKeyword) {
        let predicate = parse_expression(tokens, allocator)?;
        let if_block = parse_block(tokens, allocator)?;

        let else_block = if tokens.matches(Token::ElseKeyword) {
            if tokens.peek() == Token::IfKeyword {
                let mut stmts = AVec::new(allocator);
                stmts.push(parse_if(tokens, allocator)?)?;
                Some(Block { stmts })
            } else {
                Some(parse_block(tokens, allocator)?)
            }
        } else {
            None
        };

        Ok(Statement::IfStatement(IfStatement {
            predicate,
            if_block,
            else_block,
        }))
    } else {
        Err(PureParseError::Generic(b"(Internal) expected to match if").into())
    }
}

fn parse_statement<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Statement<A>, ParseError> {
    if tokens.peek() == Token::LeftCurly {
        let block = parse_block(tokens, allocator)?;

        // TODO: if there's a semicolon this was actually a block _expression_

        Ok(Statement::Block(block))
    } else if tokens.matches(Token::LetKeyword) {
        if let Token::Identifier(name) = tokens.peek() {
            let mut name_vec = AVec::new(allocator);
            name_vec.extend_from_slice(&name)?;

            tokens.advance();
            if !tokens.matches(Token::Equal) {
                return Err(PureParseError::Generic(b"Missing equal in declaration").into());
            }
            let expr = parse_expression(tokens, allocator)?;

            if !tokens.matches(Token::Semicolon) {
                return Err(PureParseError::Generic(b"Missing semicolon after declaration").into());
            }

            Ok(Statement::Declaration(name_vec, expr))
        } else {
            return Err(PureParseError::Generic(b"Missing identifier after let").into());
        }
    } else if tokens.peek() == Token::IfKeyword {
        Ok(parse_if(tokens, allocator)?)
    } else if tokens.matches(Token::WhileKeyword) {
        let predicate = parse_expression(tokens, allocator)?;
        let block = parse_block(tokens, allocator)?;
        Ok(Statement::WhileStatement(predicate, block))
    } else if tokens.matches(Token::BreakKeyword) {
        if !tokens.matches(Token::Semicolon) {
            return Err(PureParseError::Generic(b"Missing semicolon after break").into());
        }
        Ok(Statement::BreakStatement)
    } else if tokens.matches(Token::ContinueKeyword) {
        if !tokens.matches(Token::Semicolon) {
            return Err(PureParseError::Generic(b"Missing semicolon after continue").into());
        }
        Ok(Statement::ContinueStatement)
    } else {
        let expr = parse_expression(tokens, allocator)?;
        let stmt = match (expr, tokens.peek()) {
            (Expression::Identifier(ident), Token::Equal) => {
                tokens.advance();
                let expr = parse_expression(tokens, allocator)?;

                Statement::Assignment(ident, expr)
            }
            (expr, _) => {
                Statement::Expression(expr)
            }
        };

        if !tokens.matches(Token::Semicolon) {
            return Err(PureParseError::Generic(b"Missing semicolon after expression").into());
        }

        Ok(stmt)
    }
}

fn parse_expression<'a, A: Allocator>(
    tokens: &mut TokenStream,
    allocator: A,
) -> Result<Expression<A>, ParseError> {
    parse_equality(tokens, allocator)
}

// These values do not get a copy of the Allocator since they are GC'd, and
// the memory is handled externally.
pub enum ShimValue<A: Allocator> {
    // A variant used to replace a previous-valid value after GC
    Freed,
    // Hard-code this for now until we can declare values
    PrintFn,
    Unit,
    Bool(bool),
    I128(i128),
    F64(f64),
    // This one actually _does_ get a copy of the Allocator, but that's because
    // it doesn't have any external references.
    SString(AVec<u8, A>),
}

impl<A: Allocator> ShimValue<A> {
    fn stringify(&self, allocator: A) -> Result<AVec<u8, A>, AllocError> {
        let mut vec = AVec::new(allocator);
        match self {
            Self::I128(val) => {
                let mut val: i128 = *val;
                if val < 0 {
                    vec.push(b'-')?;
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

                vec.extend_from_slice(&slice[39 - length..39])?;
            }
            Self::F64(val) => {
                let mut buffer = [0u8; f64::FORMATTED_SIZE];
                let slice = lexical_core::write(*val, &mut buffer);
                vec.extend_from_slice(&slice)?
            }
            Self::Bool(true) => vec.extend_from_slice(b"true")?,
            Self::Bool(false) => vec.extend_from_slice(b"false")?,
            Self::Freed => vec.extend_from_slice(b"*freed*")?,
            Self::PrintFn => vec.extend_from_slice(b"<function print>")?,
            Self::Unit => vec.extend_from_slice(b"()")?,
            Self::SString(s) => vec.extend_from_slice(s)?,
        }

        Ok(vec)
    }

    fn is_truthy(&self) -> bool {
        match self {
            Self::I128(val) => *val != 0,
            Self::F64(val) => *val != 0.0,
            Self::Bool(b) => *b,
            Self::Freed => false,
            Self::PrintFn => true,
            Self::Unit => false,
            Self::SString(s) => s.len() > 0,
        }
    }
}

// A newtype which represents an index into a GC'd collection of ShimValue
#[derive(Copy, Clone)]
pub struct Id(usize);

impl Id {
    fn new(id: usize) -> Self {
        Id(id)
    }
}

struct Environment<A: Allocator> {
    map: AHashMap<AVec<u8, A>, Id, A>
}

pub struct Interpreter<'a, A: Allocator> {
    allocator: A,
    values: AVec<ShimValue<A>, A>,
    env: Environment<A>,
    // TODO: figure out how to make the ABox work like this
    print: Option<&'a mut dyn Printer>,
}

pub trait Printer {
    fn print(&mut self, text: &[u8]);
}

trait NewValue<T> {
    fn new_value(&mut self, val: T) -> Result<Id, AllocError>;
}

impl<'a, A: Allocator> NewValue<bool> for Interpreter<'a, A> {
    fn new_value(&mut self, val: bool) -> Result<Id, AllocError> {
        let id = Id::new(self.values.len());
        self.values.push(ShimValue::Bool(val))?;

        Ok(id)
    }
}

impl<'a, A: Allocator> NewValue<i128> for Interpreter<'a, A> {
    fn new_value(&mut self, val: i128) -> Result<Id, AllocError> {
        let id = Id::new(self.values.len());
        self.values.push(ShimValue::I128(val))?;

        Ok(id)
    }
}

impl<'a, A: Allocator> NewValue<f64> for Interpreter<'a, A> {
    fn new_value(&mut self, val: f64) -> Result<Id, AllocError> {
        let id = Id::new(self.values.len());
        self.values.push(ShimValue::F64(val))?;

        Ok(id)
    }
}

impl<'a, A: Allocator> NewValue<ShimValue<A>> for Interpreter<'a, A> {
    fn new_value(&mut self, val: ShimValue<A>) -> Result<Id, AllocError> {
        let id = Id::new(self.values.len());
        self.values.push(val)?;

        Ok(id)
    }
}

enum BlockExit {
    Break,
    Continue,
    Finish,
}

impl<'a, A: Allocator> Interpreter<'a, A> {
    pub fn new(allocator: A) -> Interpreter<'a, A> {
        Interpreter {
            allocator,
            values: AVec::new(allocator),
            env: Environment { map: AHashMap::new(allocator) },
            print: None,
        }
    }

    pub fn set_print_fn(&mut self, f: &'a mut dyn Printer) {
        self.print = Some(f);
    }

    pub fn print(&mut self, text: &[u8]) {
        self.print.as_mut().map(|p| p.print(text));
    }

    pub fn interpret_expression(&mut self, expr: &Expression<A>) -> Result<Id, ShimError> {
        Ok(match expr {
            Expression::Identifier(ident) => {
                let ident_slice: &[u8] = ident;
                if ident_slice == b"print" {
                    return self.new_value(ShimValue::PrintFn).map_err(|e| e.into());
                }
                // TODO: this is very dumb. AHashMap should be updated so that
                // the get method does deref-magic.
                let mut vec = AVec::new(self.allocator);
                vec.extend_from_slice(ident)?;
                let id = self
                    .env
                    .map
                    .get(&vec)
                    .map(|id| *id)
                    .ok_or_else(|| {
                        self.print(b"Could not find ");
                        self.print(&vec);
                        self.print(b" in current environment\n");
                        ShimError::Other(b"ident not found")
                    })?;

                id
            }
            Expression::IntLiteral(i) => self.new_value(*i)?,
            Expression::FloatLiteral(f) => self.new_value(*f)?,
            Expression::BoolLiteral(b) => self.new_value(*b)?,
            Expression::StringLiteral(s) => {
                let mut new_str = AVec::new(self.allocator);
                new_str.extend_from_slice(s)?;
                self.new_value(ShimValue::SString(new_str))?
            }
            Expression::Binary(op, left, right) => {
                let left = self.interpret_expression(&*left)?;
                let right = self.interpret_expression(&*right)?;

                let result = match (&self.values[left.0], &self.values[right.0]) {
                    (ShimValue::I128(a), ShimValue::I128(b)) => {
                        match op {
                            BinaryOp::Add => ShimValue::I128(a + b),
                            BinaryOp::Sub => ShimValue::I128(a - b),
                            BinaryOp::Mul => ShimValue::I128(a * b),
                            BinaryOp::Div => ShimValue::I128(a / b),
                            BinaryOp::Eq => ShimValue::Bool(a == b),
                            BinaryOp::Neq => ShimValue::Bool(a != b),
                            BinaryOp::Gt => ShimValue::Bool(a > b),
                            BinaryOp::Gte => ShimValue::Bool(a >= b),
                            BinaryOp::Lt => ShimValue::Bool(a < b),
                            BinaryOp::Lte => ShimValue::Bool(a <= b),
                        }
                    }
                    _ => {
                        self.print(b"TODO: values can't be bin-opped\n");
                        ShimValue::I128(42)
                    }
                };

                self.new_value(result)?
            }
            Expression::Call(cexpr) => {
                let func = self.interpret_expression(&cexpr.func)?;
                match self.values[func.0] {
                    ShimValue::PrintFn => {
                        let last_idx = cexpr.args.len() as isize - 1;
                        for (idx, arg) in cexpr.args.iter().enumerate() {
                            let arg = self.interpret_expression(arg)?;

                            let arg_str: AVec<u8, A> =
                                self.values[arg.0].stringify(self.allocator)?;
                            self.print(&arg_str);

                            if idx as isize != last_idx {
                                self.print(b" ");
                            }
                        }
                        self.print(b"\n");
                        self.new_value(ShimValue::Unit)?
                    }
                    _ => {
                        self.print(b"Can't call value\n");
                        self.new_value(ShimValue::I128(42))?
                    }
                }
            }
        })
    }

    fn interpret_block(&mut self, block: &Block<A>) -> Result<BlockExit, ShimError> {
        for stmt in block.stmts.iter() {
            match self.interpret_statement(stmt)? {
                Some(BlockExit::Finish) => {},
                None => {},

                // Special exit cases
                Some(BlockExit::Break) => return Ok(BlockExit::Break),
                Some(BlockExit::Continue) => return Ok(BlockExit::Continue),
            }
        }

        // TODO: return a ShimValue if the last expression didn't end with a
        // semicolon (like Rust)
        Ok(BlockExit::Finish)
    }

    fn interpret_statement(&mut self, stmt: &Statement<A>) -> Result<Option<BlockExit>, ShimError> {
        match stmt {
            Statement::Expression(expr) => {
                self.interpret_expression(expr)?;
            }
            Statement::Declaration(name, expr) => {
                let id = self.interpret_expression(expr)?;
                let name_clone: AVec<u8, A> = name.clone(self.allocator)?;
                self.env.map.insert(name_clone, id)?;
            }
            Statement::Assignment(name, expr) => {
                // TODO: this should write the value to the existing id rather
                // than writing a new id.
                let id = self.interpret_expression(expr)?;
                self.env
                    .map
                    .get_mut(&name)
                    .map(|env_id| *env_id = id)
                    .ok_or_else(|| {
                        self.print(b"Variable ");
                        self.print(&name);
                        self.print(b" has not been declared\n");
                        ShimError::Other(b"ident not found")
                    })?;
            }
            Statement::Block(block) => {
                self.interpret_block(block)?;
            }
            Statement::IfStatement(if_stmt) => {
                let predicate_result = self.interpret_expression(&if_stmt.predicate)?;
                let exit_result = if self.values[predicate_result.0].is_truthy() {
                    self.interpret_block(&if_stmt.if_block)?
                } else if let Some(else_block) = &if_stmt.else_block {
                    self.interpret_block(else_block)?
                } else {
                    BlockExit::Finish
                };

                match exit_result {
                    BlockExit::Break => return Ok(Some(BlockExit::Break)),
                    BlockExit::Continue => return Ok(Some(BlockExit::Continue)),
                    BlockExit::Finish => {},
                }
            }
            Statement::WhileStatement(predicate, block) => {
                loop {
                    let id = self.interpret_expression(&predicate)?;
                    if !self.values[id.0].is_truthy() {
                        break;
                    }
                    match self.interpret_block(&block)? {
                        BlockExit::Break => return Ok(Some(BlockExit::Finish)),
                        BlockExit::Continue | BlockExit::Finish => continue,
                    }
                }
            }
            Statement::BreakStatement => {
                return Ok(Some(BlockExit::Break))
            }
            Statement::ContinueStatement => {
                return Ok(Some(BlockExit::Continue))
            }
        }

        Ok(None)
    }

    pub fn interpret(&mut self, text: &'a [u8]) -> Result<(), ShimError> {
        let mut tokens = TokenStream::new(text);
        let script = match parse_script(&mut tokens, self.allocator) {
            Ok(script) => script,
            Err(ParseError::PureParseError(PureParseError::Generic(msg))) => {
                self.print(msg);
                self.print(b"\n");
                return Err(ParseError::PureParseError(PureParseError::Generic(msg)).into());
            }
            Err(err) => {
                self.print(b"Error parsing input");
                return Err(err.into());
            }
        };

        match self.interpret_block(&script) {
            Ok(_) => {},
            Err(ShimError::Other(msg)) => {
                self.print(b"ERROR: ");
                self.print(msg);
                self.print(b"\n");
            }
            Err(_) => {
                self.print(b"ERROR: Misc error\n");
            }
        }

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

    #[test]
    fn tokenize_float() {
        let text = b"1.1";
        let mut tokens = TokenStream::new(text);

        assert_eq!(tokens.peek(), Token::FloatLiteral(1.1));
    }

    #[test]
    fn tokenize_with_newline() {
        let text = b"print(0);\nprint(0);";
        let mut tokens = TokenStream::new(text);

        for _ in 1..=2 {
            assert!(tokens.peek() == Token::Identifier(b"print"));
            tokens.advance();
            assert!(tokens.matches(Token::LeftParen));
            assert!(tokens.peek() == Token::IntLiteral(0));
            tokens.advance();
            assert!(tokens.matches(Token::RightParen));
            assert!(tokens.matches(Token::Semicolon));
        }

        assert!(tokens.matches(Token::EOF));
    }

    #[test]
    fn tokenize_if() {
        let text = b"if";
        let mut tokens = TokenStream::new(text);

        assert!(tokens.peek() == Token::IfKeyword);
    }

    #[test]
    fn tokenize_str() {
        let text = br#"print("hi");"#;
        let mut tokens = TokenStream::new(text);

        assert!(tokens.peek() == Token::Identifier(b"print"));
        tokens.advance();
        assert!(tokens.matches(Token::LeftParen));
        assert!(tokens.peek() == Token::StringLiteral(b"hi"));
        tokens.advance();
        assert!(tokens.matches(Token::RightParen));
        assert!(tokens.matches(Token::Semicolon));
    }

    #[test]
    fn tokenize_str2() {
        let text = br#"
            let a = "foo";
            let b = a;
            print(b);
        "#;
        let mut tokens = TokenStream::new(text);

        assert!(tokens.matches(Token::LetKeyword));
        assert!(tokens.peek() == Token::Identifier(b"a"));
        tokens.advance();
        assert!(tokens.matches(Token::Equal));
        assert!(tokens.peek() == Token::StringLiteral(b"foo"));
        tokens.advance();
        assert!(tokens.matches(Token::Semicolon));

        dbg!(tokens.peek());
        assert!(tokens.matches(Token::LetKeyword));
        assert!(tokens.peek() == Token::Identifier(b"b"));
        tokens.advance();
        assert!(tokens.matches(Token::Equal));
        assert!(tokens.peek() == Token::Identifier(b"a"));
        tokens.advance();
        assert!(tokens.matches(Token::Semicolon));

        assert!(tokens.peek() == Token::Identifier(b"print"));
        tokens.advance();
        assert!(tokens.matches(Token::LeftParen));
        assert!(tokens.peek() == Token::Identifier(b"b"));
        tokens.advance();
        assert!(tokens.matches(Token::RightParen));
        assert!(tokens.matches(Token::Semicolon));
    }
}
