use crate::parse::Span;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Dot,
    DotDot,
    Bang,
    Comma,
    Colon,
    LBracket,
    RBracket,
    Plus,
    Minus,
    Slash,
    Star,
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    And,
    Or,
    Break,
    Continue,
    Struct,
    Return,
    Equal,
    DEqual,
    BangEqual,
    GT,
    Gte,
    LT,
    Lte,
    Percent,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    Semicolon,
    LSquare,
    RSquare,
    LAngle,
    RAngle,
    LCurly,
    RCurly,
    None,
    Integer(i32),
    Float(f32),
    Bool(bool),
    Identifier(Vec<u8>),
    String(Vec<u8>),
    StringInterpolationStart,
    StringInterpolationEnd,
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::LSquare => write!(f, "["),
            Token::LCurly => write!(f, "{{"),
            Token::LBracket => write!(f, "("),
            _ => write!(f, "{self:?}"),
        }
    }
}

pub struct TokenStream {
    pub(crate) idx: usize,
    pub(crate) tokens: Vec<Token>,
    // Technically we don't need to store this since we could recompute this
    // from the script when we need to show an error. This just keeps things
    // simple for now. The upper index is not inclusive, so (10,11) is 1 character.
    pub(crate) token_spans: Vec<Span>,
    pub(crate) script: Vec<u8>,
}

#[derive(Debug)]
struct LineInfo {
    start_idx: u32,
    end_idx: u32,
}

fn script_lines(script: &[u8]) -> Vec<LineInfo> {
    let mut line_info = Vec::new();
    let mut line_start_idx: u32 = 0;
    for (idx, c) in script.iter().enumerate() {
        if *c == b'\n' {
            line_info.push(LineInfo {
                start_idx: line_start_idx,
                end_idx: idx as u32,
            });
            line_start_idx = idx as u32 + 1;
        }
    }

    // If the last character was not a newline
    if line_start_idx != script.len() as u32 {
        line_info.push(LineInfo {
            start_idx: line_start_idx,
            end_idx: script.len() as u32 - 1,
        });
    }

    line_info
}

pub fn debug_u8s(data: &[u8]) -> &str {
    unsafe { std::str::from_utf8_unchecked(data) }
}

pub(crate) fn format_script_err(span: Span, script: &[u8], msg: &str) -> String {
    let script_lines = script_lines(script);
    let mut out = "".to_string();

    if script_lines.is_empty() {
        out.push_str(&format!("Error: {msg}"));
        return out;
    }

    // Find which lines the span covers
    let mut first_line: usize = 0;
    let mut last_line: usize = 0;
    for (lineno, line_info) in script_lines.iter().enumerate() {
        if line_info.start_idx <= span.start && span.start <= line_info.end_idx {
            first_line = lineno;
        }
        // The `+ 1` accounts for span.end being one past the last character
        // (e.g. at a newline or EOF position just after the line ends)
        if line_info.start_idx <= span.end && span.end <= line_info.end_idx + 1 {
            last_line = lineno;
        }
    }

    // The width of the line number field
    let lineno_width = script_lines.len().to_string().len();
    // The `gutter_size` includes everything up to the first character of the line
    let gutter_size = lineno_width + 4;
    let is_multiline = first_line != last_line;

    if !is_multiline {
        // For single-line errors, show 2 lines of context before and after
        let display_start = first_line.saturating_sub(2);
        let display_end = (last_line + 2).min(script_lines.len() - 1);

        for (lineno_0, line_info) in script_lines.iter().enumerate() {
            let lineno = lineno_0 + 1;

            if lineno_0 < display_start || lineno_0 > display_end {
                continue;
            }

            let line: String = unsafe {
                std::str::from_utf8_unchecked(
                    &script[line_info.start_idx as usize..=line_info.end_idx as usize],
                )
                .to_string()
            };

            out.push_str(&format!(
                " {:lineno_size$} | {}",
                lineno,
                line,
                lineno_size = gutter_size - 4
            ));
            if !line.ends_with("\n") {
                out.push('\n');
            }

            if lineno_0 == first_line {
                let line_span_start = span.start - line_info.start_idx;
                let line_span_end = span.end - line_info.start_idx;

                out.push_str(&" ".repeat(gutter_size));
                out.push_str(&" ".repeat(line_span_start as usize));
                out.push_str(&"^".repeat((line_span_end - line_span_start) as usize));
                out.push('\n');
            }
        }

        out.push_str(&format!("Error: {msg}"));
    } else {
        // Rust-style multiline error display
        let start_col = (span.start - script_lines[first_line].start_idx) as usize + 1;
        let end_col = (span.end - script_lines[last_line].start_idx) as usize;
        let gutter_pad = " ".repeat(lineno_width + 1);

        out.push_str(&format!("Error: {msg}\n"));
        out.push_str(&format!(
            "{gutter_pad} --> {}:{}\n",
            first_line + 1,
            start_col
        ));
        out.push_str(&format!("{gutter_pad} |\n"));

        #[allow(clippy::needless_range_loop)]
        for lineno_0 in first_line..=last_line {
            let lineno = lineno_0 + 1;
            let line_info = &script_lines[lineno_0];
            let line: String = unsafe {
                std::str::from_utf8_unchecked(
                    &script[line_info.start_idx as usize..=line_info.end_idx as usize],
                )
                .to_string()
            };

            let marker = if lineno_0 == first_line { "/" } else { "|" };

            out.push_str(&format!(
                " {:lineno_size$} | {marker} {}",
                lineno,
                line,
                lineno_size = lineno_width,
            ));
            if !line.ends_with("\n") {
                out.push('\n');
            }
        }

        // Closing line with ^
        out.push_str(&format!("{gutter_pad} | |{}^\n", "_".repeat(end_col),));
    }

    out
}

impl TokenStream {
    /**
     * Return the next token (if there are tokens remaining) without advancing the stream
     */
    pub(crate) fn peek(&self) -> Result<&Token, String> {
        if self.is_empty() {
            Ok(&Token::EOF)
        } else {
            Ok(&self.tokens[self.idx])
        }
    }

    pub(crate) fn peek_span(&self) -> Result<Span, String> {
        if self.is_empty() {
            Ok(self.token_spans[self.token_spans.len() - 1])
        } else {
            Ok(self.token_spans[self.idx])
        }
    }

    /// Return the span of the most recently consumed token
    pub(crate) fn previous_span(&self) -> Result<Span, String> {
        if self.idx > 0 {
            Ok(self.token_spans[self.idx - 1])
        } else {
            Err("No previous token".to_string())
        }
    }

    /**
     * Return the next token (if there are tokens remaining) and advance the stream
     */
    pub(crate) fn pop(&mut self) -> Result<Token, String> {
        if !self.is_empty() {
            let result = self.tokens[self.idx].clone();
            self.idx += 1;
            Ok(result)
        } else {
            Ok(Token::EOF)
        }
    }

    pub(crate) fn consume(&mut self, expected: Token) -> Result<(), String> {
        let value = self.pop()?;
        if value == expected {
            Ok(())
        } else {
            self.unadvance()?;
            Err(self.format_peek_err(&format!(
                "Expected token {:?} but found {:?}",
                expected, value
            )))
        }
    }

    pub(crate) fn advance(&mut self) -> Result<(), String> {
        if self.pop()? == Token::EOF {
            return Err(self.format_peek_err("End of token stream"));
        }
        Ok(())
    }

    pub(crate) fn unadvance(&mut self) -> Result<(), String> {
        if self.idx != 0 {
            self.idx -= 1;
            Ok(())
        } else {
            Err("Can't unadvance past beginning of token stream".to_string())
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.idx >= self.tokens.len()
    }

    pub(crate) fn format_peek_err(&self, msg: &str) -> String {
        let span = if !self.is_empty() {
            self.token_spans[self.idx]
        } else {
            Span {
                start: self.script.len() as u32 - 1,
                end: self.script.len() as u32,
            }
        };
        format_script_err(span, &self.script, msg)
    }

    pub fn spans(&self) -> Vec<Span> {
        self.token_spans.clone()
    }
}

pub fn printable_byte(b: u8) -> String {
    match char::from_u32(b as u32) {
        Some(c) if !c.is_control() => c.to_string(),
        _ => format!("\\x{:02X}", b),
    }
}

pub fn lex_identifier(text: &mut &[u8]) -> Result<Vec<u8>, String> {
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => continue,
            _ => {
                let ident = text[0..idx].to_vec();
                *text = &text[(idx - 1)..];
                return Ok(ident);
            }
        }
    }
    // End of string - consume all of text
    Ok(text.to_vec())
}

pub enum StringLexResult {
    Literal(Vec<u8>),
    Interpolation(Vec<u8>),
}

pub fn lex_string(text: &mut &[u8]) -> Result<StringLexResult, String> {
    let enclosing_char = b'"';
    *text = &text[1..];
    let mut out: Vec<u8> = Vec::new();
    let mut escape_next = false;
    for (idx, c) in text.iter().enumerate() {
        if escape_next {
            match c {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'\\' => out.push(b'\\'),
                b'"' => out.push(b'"'),
                b'(' => {
                    *text = &text[idx..];
                    return Ok(StringLexResult::Interpolation(out));
                }
                b => return Err(format!("Could not escape {:?}", printable_byte(*b))),
            }
            escape_next = false;
            continue;
        }
        if *c == enclosing_char {
            *text = &text[idx..];
            return Ok(StringLexResult::Literal(out));
        }
        match c {
            b'\\' => {
                escape_next = true;
                continue;
            }
            _ => {
                out.push(*c);
            }
        }
    }
    Err(format!(
        "No closing quote {:?} found",
        printable_byte(enclosing_char)
    ))
}

pub fn lex_number(text: &mut &[u8]) -> Result<Token, String> {
    let mut found_decimal = false;
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'0'..=b'9' => continue,
            b'.' => {
                // Check if this is a range operator (..)
                if idx + 1 < text.len() && text[idx + 1] == b'.' {
                    // This is a range operator, stop here
                    let token = if found_decimal {
                        Token::Float(unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                                let string_slice = match std::str::from_utf8(slice) {
                                    Ok(s) => s,
                                    Err(e) => return format!("Not utf-8 {:?}", e),
                                };
                                format!("Could not tokenize number '{}' {:?}", string_slice, e)
                            })?
                        })
                    } else {
                        Token::Integer(unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                                let string_slice = match std::str::from_utf8(slice) {
                                    Ok(s) => s,
                                    Err(e) => return format!("Not utf-8 {:?}", e),
                                };
                                format!("Could not tokenize number '{}' {:?}", string_slice, e)
                            })?
                        })
                    };
                    *text = &text[(idx - 1)..];
                    return Ok(token);
                }
                if found_decimal {
                    return Err("Found multiple decimals in number".to_string());
                }
                found_decimal = true;
            }
            _ => {
                let token = if found_decimal {
                    Token::Float(unsafe {
                        let slice = &text[..idx];
                        std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                            let string_slice = match std::str::from_utf8(slice) {
                                Ok(s) => s,
                                Err(e) => return format!("Not utf-8 {:?}", e),
                            };
                            format!("Could not tokenize number '{}' {:?}", string_slice, e)
                        })?
                    })
                } else {
                    Token::Integer(unsafe {
                        let slice = &text[..idx];
                        std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                            let string_slice = match std::str::from_utf8(slice) {
                                Ok(s) => s,
                                Err(e) => return format!("Not utf-8 {:?}", e),
                            };
                            format!("Could not tokenize number '{}' {:?}", string_slice, e)
                        })?
                    })
                };
                *text = &text[(idx - 1)..];
                return Ok(token);
            }
        }
    }
    // End of string - consume all of text
    let token = Token::Integer(unsafe {
        std::str::from_utf8_unchecked(text)
            .parse()
            .map_err(|e| format!("{:?}", e))?
    });
    Ok(token)
}

pub fn lex_multiline_comment_end_idx(text: &[u8]) -> Result<usize, String> {
    if text.len() < 4 {
        return Err("Text not long enough to finish multiline comment".to_string());
    }

    if text[..2] != *b"/*" {
        return Err("Multiline comment does not start with `/*`".to_string());
    }

    let mut depth = 1;
    let mut idx = 2;

    while text.len() - idx > (depth * 2) {
        if text[idx] == b'/' && text[idx + 1] == b'*' {
            depth += 1;
            idx += 2;
            continue;
        }

        if text[idx] == b'*' && text[idx + 1] == b'/' {
            depth -= 1;
            idx += 2;

            if depth == 0 {
                return Ok(idx);
            }
            continue;
        }
        idx += 1;
    }
    Err("Not enough text remaining to close multiline comment".to_string())
}

pub fn lex(text: &[u8]) -> Result<TokenStream, String> {
    let starting_len = text.len();
    let starting_text = text;
    let original_text = text;
    let mut text = text;
    let mut spans = Vec::new();
    let mut tokens = Vec::new();

    let mut braces: Vec<Token> = Vec::new();

    while !text.is_empty() {
        let c = text[0];
        let token_start_len = text.len();
        match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let ident = lex_identifier(&mut text)?;
                if ident == b"let" {
                    tokens.push(Token::Let);
                } else if ident == b"fn" {
                    tokens.push(Token::Fn);
                } else if ident == b"if" {
                    tokens.push(Token::If);
                } else if ident == b"else" {
                    tokens.push(Token::Else);
                } else if ident == b"in" {
                    tokens.push(Token::In);
                } else if ident == b"for" {
                    tokens.push(Token::For);
                } else if ident == b"while" {
                    tokens.push(Token::While);
                } else if ident == b"break" {
                    tokens.push(Token::Break);
                } else if ident == b"continue" {
                    tokens.push(Token::Continue);
                } else if ident == b"struct" {
                    tokens.push(Token::Struct);
                } else if ident == b"return" {
                    tokens.push(Token::Return);
                } else if ident == b"and" {
                    tokens.push(Token::And);
                } else if ident == b"or" {
                    tokens.push(Token::Or);
                } else if ident == b"true" {
                    tokens.push(Token::Bool(true));
                } else if ident == b"false" {
                    tokens.push(Token::Bool(false));
                } else if ident == b"None" {
                    tokens.push(Token::None);
                } else {
                    tokens.push(Token::Identifier(ident))
                }
            }
            b'0'..=b'9' => tokens.push(lex_number(&mut text)?),
            b'"' => match lex_string(&mut text)? {
                StringLexResult::Literal(s) => tokens.push(Token::String(s)),
                StringLexResult::Interpolation(s) => {
                    tokens.push(Token::String(s));
                    tokens.push(Token::StringInterpolationStart);
                    braces.push(Token::StringInterpolationStart);
                }
            },
            b'{' => {
                tokens.push(Token::LCurly);
                braces.push(Token::LCurly);
            }
            b'[' => {
                tokens.push(Token::LSquare);
                braces.push(Token::LSquare);
            }
            b'(' => {
                tokens.push(Token::LBracket);
                braces.push(Token::LBracket);
            }
            b'}' => {
                match braces.pop() {
                    Some(Token::LCurly) => (),
                    Some(b) => {
                        return Err(format_script_err(
                            Span {
                                start: (original_text.len() - text.len()) as u32,
                                end: (original_text.len() - text.len() + 1) as u32,
                            },
                            original_text,
                            &format!("Brace {b} does not match {}", c as char),
                        ));
                    }
                    None => {
                        return Err(format_script_err(
                            Span {
                                start: (original_text.len() - text.len()) as u32,
                                end: (original_text.len() - text.len() + 1) as u32,
                            },
                            original_text,
                            "No braces remaining on stack!",
                        ));
                    }
                }
                tokens.push(Token::RCurly)
            }
            b']' => {
                match braces.pop() {
                    Some(Token::LSquare) => (),
                    Some(b) => {
                        return Err(format_script_err(
                            Span {
                                start: (original_text.len() - text.len()) as u32,
                                end: (original_text.len() - text.len() + 1) as u32,
                            },
                            original_text,
                            &format!("Brace {b} does not match {}", c as char),
                        ));
                    }
                    None => {
                        return Err(format_script_err(
                            Span {
                                start: (original_text.len() - text.len()) as u32,
                                end: (original_text.len() - text.len() + 1) as u32,
                            },
                            original_text,
                            "No braces remaining on stack!",
                        ));
                    }
                }
                tokens.push(Token::RSquare);
            }
            b')' => match braces.pop() {
                Some(Token::LBracket) => tokens.push(Token::RBracket),
                Some(Token::StringInterpolationStart) => {
                    tokens.push(Token::StringInterpolationEnd);
                    match lex_string(&mut text)? {
                        StringLexResult::Literal(s) => tokens.push(Token::String(s)),
                        StringLexResult::Interpolation(s) => {
                            tokens.push(Token::String(s));
                            tokens.push(Token::StringInterpolationStart);
                            braces.push(Token::StringInterpolationStart);
                        }
                    }
                }
                Some(b) => {
                    return Err(format_script_err(
                        Span {
                            start: (original_text.len() - text.len()) as u32,
                            end: (original_text.len() - text.len() + 1) as u32,
                        },
                        original_text,
                        &format!("Brace {b} does not match {}", c as char),
                    ));
                }
                None => {
                    return Err(format_script_err(
                        Span {
                            start: (original_text.len() - text.len()) as u32,
                            end: (original_text.len() - text.len() + 1) as u32,
                        },
                        original_text,
                        "No braces remaining on stack!",
                    ));
                }
            },
            b',' => tokens.push(Token::Comma),
            b'+' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::PlusEqual);
                } else {
                    tokens.push(Token::Plus);
                }
            }
            b'*' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::StarEqual);
                } else {
                    tokens.push(Token::Star);
                }
            }
            b'%' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::PercentEqual);
                } else {
                    tokens.push(Token::Percent);
                }
            }
            b'/' => match text[1] {
                b'/' => {
                    loop {
                        text = &text[1..];
                        if text.is_empty() {
                            break;
                        }
                        if text[0] == b'\n' { break }
                    }
                    // NOTE: no token to push since this is a comment
                }
                b'*' => {
                    let idx = lex_multiline_comment_end_idx(text)?;
                    text = &text[(idx - 1)..];
                }
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::SlashEqual);
                }
                _ => tokens.push(Token::Slash),
            },
            b'-' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::MinusEqual);
                } else {
                    tokens.push(Token::Minus);
                }
            }
            b'=' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::DEqual);
                }
                _ => tokens.push(Token::Equal),
            },
            b'>' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::Gte);
                }
                _ => tokens.push(Token::GT),
            },
            b'<' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::Lte);
                }
                _ => tokens.push(Token::LT),
            },
            b';' => tokens.push(Token::Semicolon),
            b'\n' => (),
            b'\r' => (),
            b':' => tokens.push(Token::Colon),
            b'!' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::BangEqual);
                } else {
                    tokens.push(Token::Bang);
                }
            }
            b'.' => {
                if text.len() > 1 && text[1] == b'.' {
                    text = &text[1..];
                    tokens.push(Token::DotDot);
                } else {
                    tokens.push(Token::Dot);
                }
            }
            b' ' => (),
            _ => return Err(format!("Unknown character '{}'", printable_byte(c))),
        }
        text = &text[1..];
        let token_end_len = text.len();
        while tokens.len() > spans.len() {
            spans.push(Span {
                start: (starting_len - token_start_len) as u32,
                end: (starting_len - token_end_len) as u32,
            });
        }
    }
    assert_eq!(tokens.len(), spans.len(),);
    Ok(TokenStream {
        idx: 0,
        tokens,
        token_spans: spans,
        script: starting_text.to_vec(),
    })
}
