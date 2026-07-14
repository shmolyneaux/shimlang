use crate::lex::{Token, TokenStream, format_script_err, lex};
use std::ops::Add;

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Add<Span> for Span {
    type Output = Span;

    fn add(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug)]
pub struct Node<T> {
    pub data: T,
    pub span: Span,
}

impl<T> Node<T> {}

// Now redefine your types using the wrapper
pub type ExprNode = Node<Expression>;
pub type StatementNode = Node<Statement>;
pub type Ident = Vec<u8>;

#[derive(Debug)]
pub enum Primary {
    None,
    // Carried as i64 so a leading-minus fold can represent i32::MIN; it is
    // saturated to the i32 range during compilation.
    Integer(i64),
    Float(f32),
    Identifier(Vec<u8>),
    Bool(bool),
    String(Vec<u8>),
    List(Vec<ExprNode>),
    Tuple(Vec<ExprNode>),
    Expression(Box<ExprNode>),
}

#[derive(Debug)]
pub enum UnaryOp {
    Not(Box<ExprNode>),
    Negate(Box<ExprNode>),
}

#[derive(Debug)]
pub enum BinaryOp {
    Add(Box<ExprNode>, Box<ExprNode>),
    Subtract(Box<ExprNode>, Box<ExprNode>),
    Multiply(Box<ExprNode>, Box<ExprNode>),
    Divide(Box<ExprNode>, Box<ExprNode>),
    Equal(Box<ExprNode>, Box<ExprNode>),
    NotEqual(Box<ExprNode>, Box<ExprNode>),
    GT(Box<ExprNode>, Box<ExprNode>),
    Gte(Box<ExprNode>, Box<ExprNode>),
    LT(Box<ExprNode>, Box<ExprNode>),
    Lte(Box<ExprNode>, Box<ExprNode>),
    Modulus(Box<ExprNode>, Box<ExprNode>),
    In(Box<ExprNode>, Box<ExprNode>),
    Range(Box<ExprNode>, Box<ExprNode>),
}

impl BinaryOp {
    pub(crate) fn exprs(&self) -> (&ExprNode, &ExprNode) {
        match self {
            BinaryOp::Add(a, b) => (a, b),
            BinaryOp::Subtract(a, b) => (a, b),
            BinaryOp::Multiply(a, b) => (a, b),
            BinaryOp::Divide(a, b) => (a, b),
            BinaryOp::Equal(a, b) => (a, b),
            BinaryOp::NotEqual(a, b) => (a, b),
            BinaryOp::GT(a, b) => (a, b),
            BinaryOp::Gte(a, b) => (a, b),
            BinaryOp::LT(a, b) => (a, b),
            BinaryOp::Lte(a, b) => (a, b),
            BinaryOp::Modulus(a, b) => (a, b),
            BinaryOp::In(a, b) => (a, b),
            BinaryOp::Range(a, b) => (a, b),
        }
    }
}

#[derive(Debug)]
pub enum BooleanOp {
    And(Box<ExprNode>, Box<ExprNode>),
    Or(Box<ExprNode>, Box<ExprNode>),
}

impl BooleanOp {
    pub(crate) fn exprs(&self) -> (&ExprNode, &ExprNode) {
        match self {
            BooleanOp::And(a, b) => (a, b),
            BooleanOp::Or(a, b) => (a, b),
        }
    }
}

#[derive(Debug)]
pub struct Block {
    pub(crate) stmts: Vec<StatementNode>,
    pub(crate) last_expr: Option<Box<ExprNode>>,
}

impl Block {
    fn structs(&self) -> Vec<Struct> {
        let mut out = Vec::new();
        for stmt in self.stmts {
            match stmt.data {
                Statement::Struct(s) => out.push(s.clone()),
                _ => (),
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

#[derive(Debug)]
pub enum Expression {
    Primary(Primary),
    BooleanOp(BooleanOp),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    /// A chained comparison such as `a < b < c`. Holds `n` operands and
    /// `n - 1` operators (always at least two operators; single comparisons
    /// stay as `BinaryOp`). Operands are evaluated at most once, left to right,
    /// and evaluation short-circuits as soon as a comparison fails.
    Compare(Vec<ExprNode>, Vec<CompareOp>),
    Call(Box<ExprNode>, Vec<ExprNode>, Vec<(Ident, ExprNode)>),
    Index(Box<ExprNode>, Box<ExprNode>),
    Attribute(Box<ExprNode>, Vec<u8>),
    Block(Block),
    If(Box<ExprNode>, Block, Block),
    Fn(Fn),
    /// A dict literal `{ key: value, ... }`. The empty dict is spelled `{:}`.
    Dict(Vec<(ExprNode, ExprNode)>),
    /// A set literal `{ a, b, ... }`. A single-element set requires a trailing
    /// comma (`{ x, }`) and the empty set is spelled `{,}`, mirroring tuples.
    /// Parsed and represented in the AST, but not yet lowered by the compiler.
    Set(Vec<ExprNode>),
}

#[derive(Debug)]
pub struct Fn {
    pub(crate) ident: Option<Vec<u8>>,
    pub(crate) pos_args_required: Vec<Vec<u8>>,
    pub(crate) pos_args_optional: Vec<(Vec<u8>, ExprNode)>,
    pub(crate) body: Block,
}

#[derive(Debug)]
pub struct Struct {
    pub(crate) ident: Vec<u8>,
    pub(crate) members_required: Vec<Vec<u8>>,
    pub(crate) members_optional: Vec<(Vec<u8>, ExprNode)>,
    pub(crate) methods: Vec<Fn>,
}

enum Reloadable {
    Fn(Fn),
    Struct(Struct),
}

#[derive(Debug)]
pub enum CompoundOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
}

#[derive(Debug)]
pub enum Target {
    Ident(Vec<u8>),
    Tuple(Vec<Vec<u8>>),
}

#[derive(Debug)]
pub enum Statement {
    Let(Target, ExprNode),
    Assignment(Target, ExprNode),
    AttributeAssignment(ExprNode, Vec<u8>, ExprNode),
    IndexAssignment(ExprNode, ExprNode, ExprNode),
    CompoundAssignment(Vec<u8>, CompoundOp, ExprNode),
    CompoundAttributeAssignment(ExprNode, Vec<u8>, CompoundOp, ExprNode),
    CompoundIndexAssignment(ExprNode, ExprNode, CompoundOp, ExprNode),
    If(ExprNode, Block, Block),
    For(Vec<Vec<u8>>, ExprNode, Block),
    While(ExprNode, Block),
    Break,
    Continue,
    Fn(Fn),
    Struct(Struct),
    Expression(ExprNode),
    Return(Option<ExprNode>),
}

#[derive(Debug)]
pub struct Ast {
    pub(crate) block: Block,
    pub(crate) script: Vec<u8>,
}

pub struct Conditional {
    pub(crate) conditional: ExprNode,
    pub(crate) if_body: Block,
    pub(crate) else_body: Block,
}

impl Conditional {
    fn new(conditional: ExprNode, if_body: Block, else_body: Block) -> Self {
        Conditional {
            conditional,
            if_body,
            else_body,
        }
    }
}

pub fn parse_block(tokens: &mut TokenStream) -> Result<Block, String> {
    tokens.consume(Token::LCurly)?;
    let block = parse_block_inner(tokens)?;
    tokens.consume(Token::RCurly)?;

    Ok(block)
}

/// What a `{`-led expression turns out to be.
enum CurlyKind {
    Block,
    Dict,
    Set,
}

/// Decide whether the brace at the cursor (the opening `{` has *already* been
/// consumed) starts a block, a dict literal, or a set literal, without
/// consuming any tokens.
///
/// The rule is purely local and backward-compatible: `:` is otherwise illegal
/// in the grammar and a top-level comma inside braces is otherwise a parse
/// error, so the first brace-top-level `:` means a dict and the first
/// brace-top-level `,` means a set. Statement terminators (`;`), a statement
/// keyword in leading position, or reaching the closing `}` first all mean a
/// block. Nested `()`/`[]`/`{}` are skipped via depth tracking so a `:`/`,`
/// inside them (e.g. `{ d[a:b] }`) does not misfire.
fn classify_curly(tokens: &TokenStream) -> CurlyKind {
    let rest = &tokens.tokens[tokens.idx..];

    // A statement-only keyword in leading position is unambiguously a block;
    // these can never begin a dict key or set element expression. (`if`/`fn`
    // are expressions too, so they are left to the scan below.)
    if let Some(first) = rest.first() {
        match first {
            Token::Let
            | Token::While
            | Token::For
            | Token::Return
            | Token::Break
            | Token::Continue
            | Token::Struct => return CurlyKind::Block,
            _ => {}
        }
    }

    let mut depth: i32 = 0;
    for tok in rest {
        match tok {
            Token::LBracket | Token::LSquare | Token::LCurly => depth += 1,
            Token::RBracket | Token::RSquare => depth -= 1,
            Token::RCurly => {
                if depth == 0 {
                    // Closed our brace without seeing a top-level `:` or `,`.
                    return CurlyKind::Block;
                }
                depth -= 1;
            }
            Token::Colon if depth == 0 => return CurlyKind::Dict,
            Token::Comma if depth == 0 => return CurlyKind::Set,
            Token::Semicolon if depth == 0 => return CurlyKind::Block,
            _ => {}
        }
    }

    CurlyKind::Block
}

/// Parse a dict literal, assuming the opening `{` has already been consumed.
/// Consumes through the closing `}`. The empty dict is `{:}`.
fn parse_dict_literal(tokens: &mut TokenStream) -> Result<Vec<(ExprNode, ExprNode)>, String> {
    let mut pairs = Vec::new();

    // Empty dict literal: `{:}`
    if *tokens.peek()? == Token::Colon {
        tokens.advance()?;
        tokens.consume(Token::RCurly)?;
        return Ok(pairs);
    }

    loop {
        let key = parse_expression(tokens)?;
        tokens.consume(Token::Colon)?;
        let value = parse_expression(tokens)?;
        pairs.push((key, value));

        match tokens.peek()? {
            Token::RCurly => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                // Allow a trailing comma before the closing brace.
                if !tokens.is_empty() && *tokens.peek()? == Token::RCurly {
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected `,` or `}}` in dict literal, found {:?}",
                    token
                )));
            }
        }
    }

    Ok(pairs)
}

/// Parse a set literal, assuming the opening `{` has already been consumed.
/// Consumes through the closing `}`. The empty set is `{,}` and a single
/// element requires a trailing comma (`{ x, }`).
fn parse_set_literal(tokens: &mut TokenStream) -> Result<Vec<ExprNode>, String> {
    let mut items = Vec::new();

    // Empty set literal: `{,}`
    if *tokens.peek()? == Token::Comma {
        tokens.advance()?;
        tokens.consume(Token::RCurly)?;
        return Ok(items);
    }

    loop {
        items.push(parse_expression(tokens)?);

        match tokens.peek()? {
            Token::RCurly => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                // Allow a trailing comma before the closing brace.
                if !tokens.is_empty() && *tokens.peek()? == Token::RCurly {
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected `,` or `}}` in set literal, found {:?}",
                    token
                )));
            }
        }
    }

    Ok(items)
}

pub fn parse_primary(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let span = tokens.peek_span()?;
    let expr: Expression = match tokens.pop()? {
        Token::None => Expression::Primary(Primary::None),
        Token::Integer(i) => Expression::Primary(Primary::Integer(i)),
        Token::Float(f) => Expression::Primary(Primary::Float(f)),
        Token::String(s) => {
            let mut expr = Expression::Primary(Primary::String(s));

            while !tokens.is_empty() {
                match *tokens.peek()? {
                    Token::StringInterpolationStart => {
                        tokens.advance()?;
                        let value_expr = parse_expression(tokens)?;

                        // After the value to format, allow optional positional
                        // and keyword arguments that are forwarded to the value's
                        // `.format` method, e.g. `\(value, pretty=true)`.
                        let (args, kwargs) = if *tokens.peek()? == Token::Comma {
                            tokens.advance()?;
                            parse_fn_arguments(tokens, Token::StringInterpolationEnd)?
                        } else {
                            tokens.consume(Token::StringInterpolationEnd)?;
                            (Vec::new(), Vec::new())
                        };

                        // `\(value, ...)` is lowered to `value.format(...)`.
                        let format_call = Expression::Call(
                            Box::new(Node {
                                data: Expression::Attribute(
                                    Box::new(value_expr),
                                    b"format".to_vec(),
                                ),
                                span,
                            }),
                            args,
                            kwargs,
                        );

                        let token = tokens.pop()?;
                        match token {
                            Token::String(s) => {
                                expr = Expression::BinaryOp(BinaryOp::Add(
                                    Box::new(Node { data: expr, span }),
                                    Box::new(Node {
                                        data: format_call,
                                        span,
                                    }),
                                ));
                                expr = Expression::BinaryOp(BinaryOp::Add(
                                    Box::new(Node { data: expr, span }),
                                    Box::new(Node {
                                        data: Expression::Primary(Primary::String(s)),
                                        span,
                                    }),
                                ));
                            }
                            token => {
                                tokens.unadvance()?;
                                return Err(tokens.format_peek_err(&format!(
                                    "Unexpected `{:?}` after string interpolation",
                                    token
                                )));
                            }
                        }
                    }
                    _ => break,
                }
            }

            expr
        }
        Token::Bool(b) => Expression::Primary(Primary::Bool(b)),
        Token::Identifier(s) => Expression::Primary(Primary::Identifier(s)),
        Token::LCurly => {
            // A leading `{` may begin a block, a dict literal, or a set
            // literal. They are disjoint: a brace-top-level `:` means a dict, a
            // brace-top-level `,` means a set, and anything else is a block.
            // `classify_curly` scans ahead (the opening `{` is already
            // consumed) to decide which without re-parsing.
            match classify_curly(tokens) {
                CurlyKind::Block => {
                    tokens.unadvance()?;
                    let block = parse_block(tokens)?;
                    Expression::Block(block)
                }
                CurlyKind::Dict => Expression::Dict(parse_dict_literal(tokens)?),
                CurlyKind::Set => Expression::Set(parse_set_literal(tokens)?),
            }
        }
        Token::LBracket => {
            // Empty tuple: ()
            if !tokens.is_empty() && *tokens.peek()? == Token::RBracket {
                tokens.advance()?;
                Expression::Primary(Primary::Tuple(Vec::new()))
            } else {
                let first = parse_expression(tokens)?;
                match tokens.peek()? {
                    Token::RBracket => {
                        tokens.advance()?;
                        return Ok(first);
                    }
                    Token::Comma => {
                        tokens.advance()?;
                        let mut items = vec![first];
                        // Allow trailing comma for 1-tuple `(x,)`
                        if !tokens.is_empty() && *tokens.peek()? == Token::RBracket {
                            tokens.advance()?;
                        } else {
                            while !tokens.is_empty() {
                                let expr = parse_expression(tokens)?;
                                items.push(expr);
                                match tokens.peek()? {
                                    Token::RBracket => {
                                        tokens.advance()?;
                                        break;
                                    }
                                    Token::Comma => {
                                        tokens.advance()?;
                                        if !tokens.is_empty()
                                            && *tokens.peek()? == Token::RBracket
                                        {
                                            tokens.advance()?;
                                            break;
                                        }
                                        continue;
                                    }
                                    token => {
                                        return Err(tokens.format_peek_err(&format!(
                                            "Expected comma or `)` in tuple, found {:?}",
                                            token
                                        )));
                                    }
                                }
                            }
                        }
                        Expression::Primary(Primary::Tuple(items))
                    }
                    token => {
                        return Err(tokens.format_peek_err(&format!(
                            "Expected `)` or `,` after expression in `(`, found {:?}",
                            token
                        )));
                    }
                }
            }
        }
        Token::LSquare => {
            let items = parse_arguments(tokens, Token::RSquare)?;
            // TODO: fix span here
            Expression::Primary(Primary::List(items))
        }
        Token::If => {
            tokens.unadvance()?;
            let (cond, _cond_span) = parse_conditional(tokens)?;
            Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body)
        }
        Token::Fn => {
            tokens.unadvance()?;
            let f = parse_function(tokens)?;
            Expression::Fn(f)
        }
        token => {
            tokens.unadvance()?;
            return Err(tokens.format_peek_err(&format!(
                "Unexpected `{:?}` in parse_primary",
                // TODO: should display the exact character like `[` rather than name line `SemiColon`
                token
            )));
        }
    };
    Ok(Node {
        data: expr,
        span,
    })
}

pub fn parse_arguments(
    tokens: &mut TokenStream,
    closing_token: Token,
) -> Result<Vec<ExprNode>, String> {
    let mut args = Vec::new();
    if !tokens.is_empty() && *tokens.peek()? == closing_token {
        tokens.advance()?;
        return Ok(args);
    }
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;
        args.push(expr);

        match tokens.peek()? {
            token if *token == closing_token => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                if !tokens.is_empty() && *tokens.peek()? == closing_token {
                    // Exit when there's a trailing comma
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected comma or closing bracket, found {:?}",
                    token
                )));
            }
        }
    }
    Ok(args)
}

#[allow(clippy::type_complexity)]
pub fn parse_fn_arguments(
    tokens: &mut TokenStream,
    closing_token: Token,
) -> Result<(Vec<ExprNode>, Vec<(Ident, ExprNode)>), String> {
    let mut args = Vec::new();
    let mut kwargs = Vec::new();
    if !tokens.is_empty() && *tokens.peek()? == closing_token {
        tokens.advance()?;
        return Ok((args, kwargs));
    }
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;

        if *tokens.peek()? == Token::Equal {
            if let Expression::Primary(Primary::Identifier(ident)) = expr.data {
                tokens.advance()?;
                kwargs.push((ident, parse_expression(tokens)?));
            } else {
                return Err(
                    tokens.format_peek_err("Expected ident before `=` in fn args")
                );
            }
        } else {
            if !kwargs.is_empty() {
                return Err(tokens.format_peek_err("Positional arguments can't appear after keyword arguments"));
            }
            args.push(expr);
        }

        match tokens.peek()? {
            token if *token == closing_token => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                if !tokens.is_empty() && *tokens.peek()? == closing_token {
                    // Exit when there's a trailing comma
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected comma or closing bracket, found {:?}",
                    token
                )));
            }
        }
    }
    Ok((args, kwargs))
}

pub fn parse_call(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        // A `(` or `[` at the start of a new line begins a fresh statement
        // (a parenthesized expression or list literal); it must not be read as
        // a call/index on the previous line's value. A leading `.` always
        // continues the chain so method chains can span multiple lines.
        match *tokens.peek()? {
            Token::LBracket if !tokens.peek_starts_new_line() => {
                tokens.advance()?;
                let args = parse_fn_arguments(tokens, Token::RBracket)?;
                let end_span = tokens.previous_span()?;
                expr = Node {
                    span: expr.span + end_span,
                    data: Expression::Call(Box::new(expr), args.0, args.1),
                };
            }
            Token::LSquare if !tokens.peek_starts_new_line() => {
                tokens.advance()?;
                let index_expr = parse_expression(tokens)?;
                tokens.consume(Token::RSquare)?;
                let end_span = tokens.previous_span()?;
                let start = expr.span;
                expr = Node {
                    data: Expression::Index(Box::new(expr), Box::new(index_expr)),
                    span: start + end_span,
                };
            }
            Token::Dot => {
                tokens.advance()?;
                let ident_span = tokens.peek_span()?;
                let ident = match tokens.pop()? {
                    Token::Identifier(ident) => ident.clone(),
                    token => {
                        return Err(tokens.format_peek_err(&format!(
                            "Expected ident after dot, found {:?}",
                            token
                        )));
                    }
                };
                expr = Node {
                    data: Expression::Attribute(Box::new(expr), ident),
                    span: span + ident_span,
                };
            }
            _ => return Ok(expr),
        }
    }

    Ok(expr)
}

pub fn parse_logical_or(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_logical_and(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Or => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BooleanOp(BooleanOp::Or(
                        Box::new(expr),
                        Box::new(parse_logical_and(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_logical_and(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_range(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::And => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BooleanOp(BooleanOp::And(
                        Box::new(expr),
                        Box::new(parse_range(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_range(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_equality(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::DotDot => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Range(
                        Box::new(expr),
                        Box::new(parse_equality(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

/// Build a single relational `BinaryOp` node from an operator token.
fn build_relational(op: &Token, a: ExprNode, b: ExprNode, span: Span) -> ExprNode {
    let data = match op {
        Token::GT => Expression::BinaryOp(BinaryOp::GT(Box::new(a), Box::new(b))),
        Token::Gte => Expression::BinaryOp(BinaryOp::Gte(Box::new(a), Box::new(b))),
        Token::LT => Expression::BinaryOp(BinaryOp::LT(Box::new(a), Box::new(b))),
        Token::Lte => Expression::BinaryOp(BinaryOp::Lte(Box::new(a), Box::new(b))),
        _ => unreachable!("build_relational called with non-relational token"),
    };
    Node { data, span }
}

fn relational_compare_op(op: &Token) -> CompareOp {
    match op {
        Token::GT => CompareOp::Gt,
        Token::Gte => CompareOp::Gte,
        Token::LT => CompareOp::Lt,
        Token::Lte => CompareOp::Lte,
        _ => unreachable!("relational_compare_op called with non-relational token"),
    }
}

pub fn parse_comparison(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_term(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::GT | Token::Gte | Token::LT | Token::Lte => {
                // Collect a run of relational operators so that `a < b < c`
                // becomes a single chained comparison rather than `(a < b) < c`.
                let mut operands = vec![expr];
                let mut ops: Vec<Token> = Vec::new();
                while matches!(
                    tokens.peek()?,
                    Token::GT | Token::Gte | Token::LT | Token::Lte
                ) {
                    ops.push(tokens.pop()?);
                    operands.push(parse_term(tokens)?);
                }

                expr = if ops.len() == 1 {
                    let b = operands.pop().unwrap();
                    let a = operands.pop().unwrap();
                    build_relational(&ops[0], a, b, span)
                } else {
                    let compare_ops = ops.iter().map(relational_compare_op).collect();
                    Node {
                        data: Expression::Compare(operands, compare_ops),
                        span,
                    }
                };
            }
            Token::In => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::In(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_factor(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_unary(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Star => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Multiply(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span,
                };
            }
            Token::Slash => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Divide(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span,
                };
            }
            Token::Percent => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Modulus(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_equality(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_comparison(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::DEqual | Token::BangEqual => {
                // Collect a run of equality operators so that `a == b == c`
                // becomes a single chained comparison evaluated left to right.
                let mut operands = vec![expr];
                let mut ops: Vec<Token> = Vec::new();
                while matches!(tokens.peek()?, Token::DEqual | Token::BangEqual) {
                    ops.push(tokens.pop()?);
                    operands.push(parse_comparison(tokens)?);
                }

                expr = if ops.len() == 1 {
                    let b = operands.pop().unwrap();
                    let a = operands.pop().unwrap();
                    let data = match ops[0] {
                        Token::DEqual => {
                            Expression::BinaryOp(BinaryOp::Equal(Box::new(a), Box::new(b)))
                        }
                        _ => Expression::BinaryOp(BinaryOp::NotEqual(Box::new(a), Box::new(b))),
                    };
                    Node { data, span }
                } else {
                    let compare_ops = ops
                        .iter()
                        .map(|op| match op {
                            Token::DEqual => CompareOp::Eq,
                            _ => CompareOp::Ne,
                        })
                        .collect();
                    Node {
                        data: Expression::Compare(operands, compare_ops),
                        span,
                    }
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_term(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_factor(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Plus => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Add(
                        Box::new(expr),
                        Box::new(parse_factor(tokens)?),
                    )),
                    span,
                };
            }
            Token::Minus => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Subtract(
                        Box::new(expr),
                        Box::new(parse_factor(tokens)?),
                    )),
                    span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_unary(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let span = tokens.peek_span()?;
    match tokens.peek()? {
        Token::Bang => {
            tokens.advance()?;
            let expr = parse_unary(tokens)?;
            Ok(Node {
                data: Expression::UnaryOp(UnaryOp::Not(Box::new(expr))),
                span,
            })
        }
        Token::Minus => {
            tokens.advance()?;
            let expr = parse_unary(tokens)?;
            // Fold a minus applied directly to an integer literal into the
            // literal itself. This is what lets `-2147483648` (i32::MIN) be
            // written, since its magnitude does not fit in an i32 on its own.
            if let Expression::Primary(Primary::Integer(v)) = expr.data {
                Ok(Node {
                    data: Expression::Primary(Primary::Integer(-v)),
                    span,
                })
            } else {
                Ok(Node {
                    data: Expression::UnaryOp(UnaryOp::Negate(Box::new(expr))),
                    span,
                })
            }
        }
        _ => parse_call(tokens),
    }
}

pub fn parse_conditional(tokens: &mut TokenStream) -> Result<(Conditional, Span), String> {
    let start_span = tokens.peek_span()?;
    tokens.consume(Token::If)?;
    let conditional = parse_expression(tokens)?;

    let if_body = parse_block(tokens)?;

    let else_body = if !tokens.is_empty() && *tokens.peek()? == Token::Else {
        tokens.advance()?;
        if *tokens.peek()? == Token::If {
            let (elseif, elseif_span) = parse_conditional(tokens)?;
            Block {
                stmts: Vec::new(),
                last_expr: Some(Box::new(ExprNode {
                    data: Expression::If(
                        Box::new(elseif.conditional),
                        elseif.if_body,
                        elseif.else_body,
                    ),
                    span: elseif_span,
                })),
            }
        } else {
            parse_block(tokens)?
        }
    } else {
        Block {
            stmts: Vec::new(),
            last_expr: None,
        }
    };

    // Get the end span (previous token, which is the closing curly)
    let end_span = tokens.previous_span()?;

    Ok((
        Conditional::new(conditional, if_body, else_body),
        start_span + end_span,
    ))
}

pub fn parse_expression(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    parse_logical_or(tokens)
}

pub fn parse_ast(tokens: &mut TokenStream) -> Result<Ast, String> {
    let block = parse_block_inner(tokens)?;
    Ok(Ast {
        block,
        script: tokens.script.clone(),
    })
}

pub fn parse_function(tokens: &mut TokenStream) -> Result<Fn, String> {
    tokens.consume(Token::Fn)?;
    let ident = match tokens.peek()? {
        Token::Identifier(ident) => {
            let ident = ident.clone();
            tokens.advance()?;
            Some(ident)
        }
        Token::LBracket => None,
        token => {
            return Err(
                tokens.format_peek_err(&format!("Expected ident after fn, found {:?}", token))
            );
        }
    };
    tokens.consume(Token::LBracket)?;

    let mut params = Vec::new();
    let mut optional_params = Vec::new();
    while !tokens.is_empty() {
        let ident = match tokens.peek()? {
            Token::Identifier(ident) => ident.clone(),
            Token::RBracket => break,
            token => {
                return Err(tokens
                    .format_peek_err(&format!("Expected ident for fn params, found {:?}", token)));
            }
        };
        tokens.advance()?;

        let expr = if *tokens.peek()? == Token::Equal {
            tokens.advance()?;
            Some(parse_expression(tokens)?)
        } else {
            if !optional_params.is_empty() {
                return Err(
                    tokens.format_peek_err("No required arguments after optional")
                );
            }
            None
        };

        if let Some(expr) = expr {
            optional_params.push((ident, expr));
        } else {
            params.push(ident);
        }

        if *tokens.peek()? != Token::Comma {
            break;
        }
        tokens.advance()?;
    }
    tokens.consume(Token::RBracket)?;

    let body = parse_block(tokens)?;

    Ok(Fn {
        ident,
        pos_args_required: params,
        pos_args_optional: optional_params,
        body,
    })
}

/// Consume a statement terminator. An explicit `;` is consumed; otherwise a
/// statement is terminated implicitly by the next token starting a new line, a
/// closing `}`, or end of input. A `;` is still required to separate multiple
/// statements on the same line.
fn consume_statement_terminator(tokens: &mut TokenStream) -> Result<(), String> {
    match tokens.peek()? {
        Token::Semicolon => {
            tokens.pop()?;
            Ok(())
        }
        Token::RCurly | Token::EOF => Ok(()),
        _ if tokens.is_empty() || tokens.peek_starts_new_line() => Ok(()),
        token => Err(tokens.format_peek_err(&format!(
            "Expected `;` or newline to end statement, found {:?}",
            token
        ))),
    }
}

pub fn parse_block_inner(tokens: &mut TokenStream) -> Result<Block, String> {
    let mut stmts = Vec::new();
    let mut last_expr: Option<Box<ExprNode>> = None;

    while !tokens.is_empty() && *tokens.peek()? != Token::RCurly {
        let start_span = tokens.peek_span()?;
        let stmt = if *tokens.peek()? == Token::Let {
            tokens.advance()?;
            if tokens.is_empty() {
                return Err(format_script_err(
                    start_span,
                    &tokens.script,
                    "No token found after let",
                ));
            }
            let target = match tokens.pop()? {
                Token::Identifier(ident) => Target::Ident(ident.clone()),
                Token::LBracket => {
                    let mut idents = Vec::new();
                    while *tokens.peek()? != Token::RBracket {
                        let token = tokens.pop()?;
                        if let Token::Identifier(ident) = token {
                            idents.push(ident);
                        } else {
                            return Err(tokens
                                .format_peek_err(&format!("Expected ident in let tuple, found {:?}", token)));
                        }
                        if *tokens.peek()? == Token::Comma {
                            tokens.advance()?;
                            continue;
                        }
                    }
                    tokens.consume(Token::RBracket)?;
                    Target::Tuple(idents)
                }
                token => {
                    tokens.unadvance()?;
                    return Err(tokens
                        .format_peek_err(&format!("Expected ident after let, found {:?}", token)));
                }
            };

            match tokens.pop()? {
                Token::Equal => (),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!(
                        "Expected = after `let ident`, found {:?}",
                        token
                    )));
                }
            }

            let expr = parse_expression(tokens)?;
            let end_span = tokens.previous_span()?;
            consume_statement_terminator(tokens)?;

            StatementNode {
                data: Statement::Let(target, expr),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Fn
            && matches!(tokens.tokens.get(tokens.idx + 1), Some(Token::Identifier(_)))
        {
            // `fn name(...) { ... }` is a named function declaration. A bare
            // `fn(...) { ... }` (no name) is an anonymous closure expression,
            // so it falls through to the general expression path below.
            let fn_result = parse_function(tokens)?;
            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::Fn(fn_result),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::If {
            let (cond, cond_span) = parse_conditional(tokens)?;

            // Do we treat this as an expression or statement?
            if *tokens.peek()? == Token::RCurly {
                // If the next token is an RCurly, it means that this is the closing curly of the block
                let expr = Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body);
                last_expr = Some(Box::new(ExprNode {
                    data: expr,
                    span: cond_span,
                }));
                break;
            } else {
                StatementNode {
                    data: Statement::If(cond.conditional, cond.if_body, cond.else_body),
                    span: cond_span,
                }
            }
        } else if *tokens.peek()? == Token::For {
            tokens.advance()?;
            let mut idents: Vec<Vec<u8>> = vec![match tokens.pop()? {
                Token::Identifier(ident) => ident,
                token => {
                    tokens.unadvance()?;
                    return Err(tokens
                        .format_peek_err(&format!("Expected ident after for, found {:?}", token)));
                }
            }];
            while *tokens.peek()? == Token::Comma {
                tokens.consume(Token::Comma)?;
                idents.push(match tokens.pop()? {
                    Token::Identifier(ident) => ident,
                    token => {
                        tokens.unadvance()?;
                        return Err(tokens
                            .format_peek_err(&format!("Expected ident after comma, found {:?}", token)));
                    }
                });
            }
            tokens.consume(Token::In)?;
            let expr = parse_expression(tokens)?;
            let body = parse_block(tokens)?;

            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::For(idents, expr, body),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::While {
            tokens.advance()?;
            let conditional = parse_expression(tokens)?;
            let loop_body = parse_block(tokens)?;

            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::While(conditional, loop_body),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Break {
            tokens.advance()?;
            let end_span = tokens.previous_span()?;
            consume_statement_terminator(tokens)?;
            StatementNode {
                data: Statement::Break,
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Continue {
            tokens.advance()?;
            let end_span = tokens.previous_span()?;
            consume_statement_terminator(tokens)?;
            StatementNode {
                data: Statement::Continue,
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Struct {
            tokens.advance()?;
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => ident.clone(),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!(
                        "Expected ident after struct, found {:?}",
                        token
                    )));
                }
            };
            tokens.consume(Token::LCurly)?;

            let mut members = Vec::new();
            let mut optional_members = Vec::new();
            while !tokens.is_empty() {
                match tokens.pop()? {
                    Token::Identifier(ident) => {
                        if *tokens.peek()? == Token::Equal {
                            tokens.advance()?;
                            let expr = parse_expression(tokens)?;
                            optional_members.push((ident.clone(), expr));
                        } else {
                            if !optional_members.is_empty() {
                                return Err(tokens.format_peek_err("Required members not allowed after optional members"));
                            }
                            members.push(ident.clone());
                        }
                        if *tokens.peek()? != Token::Comma {
                            break;
                        }
                        // Consume the comma
                        tokens.advance()?;
                    }
                    Token::RCurly | Token::Fn => {
                        tokens.unadvance()?;
                        break;
                    }
                    token => {
                        tokens.unadvance()?;
                        return Err(tokens.format_peek_err(&format!(
                            "Expected member list after struct, found {:?}",
                            token
                        )));
                    }
                };
            }

            let mut methods = Vec::new();
            while !tokens.is_empty() {
                match tokens.peek()? {
                    Token::Fn => {
                        methods.push(parse_function(tokens)?);
                    }
                    Token::RCurly => {
                        break;
                    }
                    token => {
                        return Err(tokens.format_peek_err(&format!(
                            "Unexpected token during method parsing {:?}",
                            token
                        )));
                    }
                }
            }
            let end_span = tokens.peek_span()?;
            tokens.consume(Token::RCurly)?;

            StatementNode {
                data: Statement::Struct(Struct {
                    ident,
                    members_required: members,
                    members_optional: optional_members,
                    methods,
                }),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Return {
            tokens.advance()?;
            // A bare `return` has no value when followed by a statement
            // terminator (`;`, a newline, `}`, or end of input); otherwise the
            // value expression must begin on the same line.
            let is_bare = matches!(tokens.peek()?, Token::Semicolon | Token::RCurly)
                || tokens.is_empty()
                || tokens.peek_starts_new_line();
            if is_bare {
                let end_span = tokens.previous_span()?;
                consume_statement_terminator(tokens)?;
                StatementNode {
                    data: Statement::Return(None),
                    span: start_span + end_span,
                }
            } else {
                let expr = parse_expression(tokens)?;
                let end_span = tokens.previous_span()?;
                consume_statement_terminator(tokens)?;
                StatementNode {
                    data: Statement::Return(Some(expr)),
                    span: start_span + end_span,
                }
            }
        } else {
            let expr = parse_expression(tokens)?;
            if tokens.is_empty() {
                last_expr = Some(Box::new(expr));
                break;
            }

            // A bare block expression used as a statement does not require a
            // trailing semicolon, mirroring if/while/for statements.
            let espan = expr.span;
            let is_block_expr = matches!(expr.data, Expression::Block(_));

            match tokens.peek()? {
                Token::RCurly => {
                    last_expr = Some(Box::new(expr));
                    break;
                }
                Token::Semicolon => {
                    let end_span = tokens.peek_span()?;
                    tokens.pop()?;
                    StatementNode {
                        data: Statement::Expression(expr),
                        span: start_span + end_span,
                    }
                }
                Token::Equal => {
                    tokens.pop()?;
                    match expr.data {
                        Expression::Primary(Primary::Identifier(ident)) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::Assignment(Target::Ident(ident.clone()), expr_to_assign),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Primary(Primary::Tuple(maybe_idents)) => {
                            let mut idents = Vec::new();
                            for maybe_ident in maybe_idents.into_iter() {
                                match maybe_ident.data {
                                    Expression::Primary(Primary::Identifier(ident)) => idents.push(ident),
                                    tuple_item => {
                                        return Err(format_script_err(
                                            maybe_ident.span,
                                            &tokens.script,
                                            &format!("Can't unpack tuple to non-ident {:?}", tuple_item),
                                        ));
                                    }
                                }
                            }
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::Assignment(Target::Tuple(idents), expr_to_assign),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Attribute(expr, ident) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::AttributeAssignment(
                                    *expr,
                                    ident.clone(),
                                    expr_to_assign,
                                ),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Index(expr, index_expr) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::IndexAssignment(
                                    *expr,
                                    *index_expr,
                                    expr_to_assign,
                                ),
                                span: start_span + end_span,
                            }
                        }
                        expr_data => {
                            return Err(format_script_err(
                                expr.span,
                                &tokens.script,
                                &format!("Can't assign to {:?}", expr_data),
                            ));
                        }
                    }
                }
                Token::PlusEqual
                | Token::MinusEqual
                | Token::StarEqual
                | Token::SlashEqual
                | Token::PercentEqual => {
                    let op = match tokens.pop()? {
                        Token::PlusEqual => CompoundOp::Add,
                        Token::MinusEqual => CompoundOp::Subtract,
                        Token::StarEqual => CompoundOp::Multiply,
                        Token::SlashEqual => CompoundOp::Divide,
                        Token::PercentEqual => CompoundOp::Modulus,
                        _ => unreachable!(),
                    };
                    match expr.data {
                        Expression::Primary(Primary::Identifier(ident)) => {
                            let rhs = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::CompoundAssignment(ident.clone(), op, rhs),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Attribute(obj_expr, ident) => {
                            let rhs = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::CompoundAttributeAssignment(
                                    *obj_expr,
                                    ident.clone(),
                                    op,
                                    rhs,
                                ),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Index(obj_expr, index_expr) => {
                            let rhs = parse_expression(tokens)?;
                            let end_span = tokens.previous_span()?;
                            consume_statement_terminator(tokens)?;
                            StatementNode {
                                data: Statement::CompoundIndexAssignment(
                                    *obj_expr,
                                    *index_expr,
                                    op,
                                    rhs,
                                ),
                                span: start_span + end_span,
                            }
                        }
                        expr_data => {
                            return Err(format_script_err(
                                expr.span,
                                &tokens.script,
                                &format!("Can't use compound assignment on {:?}", expr_data),
                            ));
                        }
                    }
                }
                token => {
                    // A newline terminates a bare expression statement, just
                    // like a `;`. A bare block also needs no terminator. Two
                    // expressions on the same line without a `;` is an error.
                    if is_block_expr || tokens.peek_starts_new_line() {
                        StatementNode {
                            data: Statement::Expression(expr),
                            span: start_span + espan,
                        }
                    } else {
                        let next_span = tokens.peek_span()?;
                        return Err(format_script_err(
                            expr.span + next_span,
                            &tokens.script,
                            &format!(
                                "Expected `;` or newline after expression statement, found {:?}",
                                token
                            ),
                        ));
                    }
                }
            }
        };

        stmts.push(stmt);
    }

    Ok(Block { stmts, last_expr })
}

pub fn ast_from_text(text: &[u8]) -> Result<Ast, String> {
    let mut tokens = lex(text)?;
    parse_ast(&mut tokens)
}
