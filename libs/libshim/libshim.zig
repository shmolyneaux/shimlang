const build_options = @import("build_options");
const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// Limited precision version numbers have never been a problem for anyone /s
const VersionInfo = struct {
    major: u8,
    minor: u8,
    patch: u8,
    release_info: []const u8,

    pub fn version_string(comptime self: VersionInfo) []const u8 {
        return self.release_info;
    }
};

pub const version: VersionInfo = .{
    .major = 0,
    .minor = 0,
    .patch = 0,
    .release_info = "unreleased",
};

// Static parser types

const File = struct {
    shebang: ArrayList(u8),
    ast: Ast,
};

const Ast = struct {
    allocator: *Allocator,
    stmts: ArrayList(Statement),

    pub fn deinit(self: Ast) void {
        for (self.stmts.items) |stmt| {
            stmt.deinit(self.allocator);
        }
        self.stmts.deinit();
    }

    pub fn format(
        self: Ast,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.writeAll("AST([");

        var first = true;
        for (self.stmts.items) |stmt| {
            if (!first) {
                try writer.writeAll(",");
            }
            try writer.print("{}", .{stmt});
            first = false;
        }

        try writer.writeAll("])");
    }
};

const Block = struct {};

const IfStatement = struct { predicate: Expression, if_block: Block, else_block: ?Block };

// TODO: Multiple sorts of statements (if, for, use, while, etc.)
const StatementTag = enum { expression_statement, if_statement, assignment_statement, pretend_statement };

const Statement = union(StatementTag) {
    expression_statement: Expression,
    if_statement: IfStatement,
    assignment_statement: struct { obj: ?Expression, name: []const u8, value: Expression },
    // Ignore this for now, I just need a placeholder
    pretend_statement: bool,

    pub fn deinit(self: Statement, allocator: *Allocator) void {
        switch (self) {
            StatementTag.expression_statement => |stmt| stmt.deinit(allocator),
            else => return,
        }
    }

    pub fn format(
        self: Statement,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            StatementTag.expression_statement => |stmt| try writer.print("EXPR({})", .{stmt}),
            StatementTag.if_statement => |stmt| try writer.print("IF({})", .{stmt}),
            StatementTag.pretend_statement => try writer.print("PRETEND()", .{}),
            // TODO: lies
            StatementTag.assignment_statement => unreachable,
        }
    }
};

const UnaryOperator = enum {
    UnaryNegation,
    UnaryNot,
};
const UnaryExpr = struct {
    expr: *Expression,
    op: UnaryOperator,
};

const BinaryOperator = enum {
    BinaryAdd,
    BinarySub,
    BinaryMul,
    BinaryDiv,
    BinaryDoubleEqual,
    BinaryBangEqual,
    BinaryGreaterThanEqual,
    BinaryLessThanEqual,
    BinaryLessThan,
    BinaryGreaterThan,
};

const BinaryExpr = struct {
    left: *Expression,
    op: BinaryOperator,
    right: *Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: BinaryExpr, allocator: *Allocator) void {
        self.left.deinit(allocator);
        self.right.deinit(allocator);
        allocator.destroy(self.left);
        allocator.destroy(self.right);
    }
};

const LogicalExpr = struct {
    left: *Expression,
    op: enum { LogicalAnd, LogicalOr },
    right: *Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: LogicalExpr, allocator: *Allocator) void {
        self.left.deinit(allocator);
        self.right.deinit(allocator);
        allocator.destroy(self.left);
        allocator.destroy(self.right);
    }
};

const CallExpr = struct {
    left: *Expression,
    args: ?*Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: CallExpr, allocator: *Allocator) void {
        self.left.deinit(allocator);
        allocator.destroy(self.left);
        if (self.args) |args| {
            args.deinit(allocator);
            allocator.destroy(args);
        }
    }
};

const ExpressionTag = enum {
    int_literal,
    string_literal,
    bool_literal,
    unary,
    binary,
    logical,
    identifier,
    call,
};
const Expression = union(ExpressionTag) {
    int_literal: i128,
    string_literal: []const u8,
    bool_literal: bool,
    unary: UnaryExpr,
    binary: BinaryExpr,
    logical: LogicalExpr,
    identifier: []const u8,
    call: CallExpr,

    pub fn deinit(self: Expression, allocator: *Allocator) void {
        switch (self) {
            ExpressionTag.int_literal => |_| {},
            ExpressionTag.string_literal => |_| {},
            ExpressionTag.bool_literal => |_| {},
            ExpressionTag.unary => |_| {},
            ExpressionTag.identifier => |ident| allocator.free(ident),
            ExpressionTag.logical => |lexpr| lexpr.deinit(allocator),
            ExpressionTag.binary => |bexpr| bexpr.deinit(allocator),
            ExpressionTag.call => |cexpr| cexpr.deinit(allocator),
        }
    }

    pub fn format(
        self: Expression,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = self;
        _ = fmt;
        _ = options;

        switch (self) {
            .int_literal => |val| try writer.print("INT({})", .{val}),
            .string_literal => |val| try writer.print("'{s}'", .{val}),
            .bool_literal => |val| try writer.print("{}", .{val}),
            .unary => |val| {
                switch (val.op) {
                    .UnaryNegation => try writer.print("NEG({})", .{val.expr.*}),
                    .UnaryNot => try writer.print("BANG({})", .{val.expr.*}),
                }
            },
            .binary => |val| {
                switch (val.op) {
                    .BinaryAdd => try writer.print("ADD({}, {})", .{ val.left.*, val.right.* }),
                    .BinarySub => try writer.print("SUB({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryMul => try writer.print("MUL({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryDiv => try writer.print("DIV({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryDoubleEqual => try writer.print("DE({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryBangEqual => try writer.print("BE({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryGreaterThanEqual => try writer.print("GTE({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryLessThanEqual => try writer.print("LTE({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryLessThan => try writer.print("LT({}, {})", .{ val.left.*, val.right.* }),
                    .BinaryGreaterThan => try writer.print("GT({}, {})", .{ val.left.*, val.right.* }),
                }
            },
            // .logical => |val| try writer.print("{}", .{val}),
            .identifier => |val| {
                _ = val;
                //std.log.info("Printing ident", .{});
                //@breakpoint();
                // std.log.info("Printing ident: {s}", .{val});
                try writer.print("IDENT({s})", .{val});
            },
            .call => |val| {
                try writer.print("CALL({}, ", .{val.left.*});
                if (val.args) |args| {
                    try writer.print("{}", .{args});
                }
                try writer.print(")", .{});
            },
            else => try writer.print("SOMETHING_ELSE", .{}),
        }
    }
};

const IntLiteral = struct {
    // TODO: bigint
    val: u8,
};

const OperandTag = enum {
    add,
    sub,
};

const Operand = union(OperandTag) {
    add: [2]u8,
    sub: [2]u8,
};

// Runtime types

const ShimUnit = struct {};

const ValueTag = enum {
    // Fixed length ints
    // shim_u8,
    // shim_u16,
    // shim_u32,
    // shim_u64,
    // shim_u128,
    // shim_i8,
    // shim_i16,
    // shim_i32,
    // shim_i64,
    shim_i128,

    shim_bool,

    // Arbitrary sized ints
    // shim_biguint,
    // shim_bigint,

    // String
    shim_str,

    // This is the value returned by statements
    shim_unit,

    shim_error,

    // ... everything else
    shim_obj,
};
const ShimValue = union(ValueTag) {
    shim_i128: i128,
    shim_str: bool,
    shim_bool: bool,
    shim_unit: ShimUnit,
    shim_error: ShimUnit,
    shim_obj: ShimUnit,
};

// Functions!

pub fn run_text(allocator: *Allocator, text: []const u8) !void {
    var ast = try parse_text(allocator, text);
    defer ast.deinit();

    _ = interpret_ast(ast);
}

const TokenTag = enum {
    string_literal,
    int_literal,
    float_literal,
    left_paren,
    right_paren,
    left_curly,
    right_curly,
    left_angle,
    right_angle,
    left_square,
    right_square,
    dot,
    plus,
    minus,
    sub,
    star,
    double_star,
    slash,
    identifier,
    colon,
    double_colon,
    semicolon,
    arrow,
    double_equal,
    equal,
    bang,
    bang_equal,
    comma,
    gte,
    lte,
    // Keywords
    and_keyword,
    as_keyword,
    break_keyword,
    continue_keyword,
    elif_keyword,
    else_keyword,
    enum_keyword,
    fn_keyword,
    for_keyword,
    if_keyword,
    or_keyword,
    return_keyword,
    struct_keyword,
    use_keyword,
    while_keyword,
    true_keyword,
    false_keyword,
};

const TokenInfo = union(TokenTag) {
    // TODO: for now this is a copy of the input
    string_literal: struct { text: []const u8 },
    int_literal: struct { value: i128 },
    float_literal: struct { value: f128 },
    left_paren,
    right_paren,
    left_curly,
    right_curly,
    left_angle,
    right_angle,
    left_square,
    right_square,
    dot,
    plus,
    minus,
    sub,
    star,
    double_star,
    slash,
    // TODO: for now this is a copy of the input
    identifier: struct { text: []const u8 },
    colon,
    double_colon,
    semicolon,
    arrow,
    double_equal,
    equal,
    bang,
    bang_equal,
    comma,
    gte,
    lte,
    // TODO: plus_equals, etc.
    // Keywords
    and_keyword,
    as_keyword,
    break_keyword,
    continue_keyword,
    elif_keyword,
    else_keyword,
    enum_keyword,
    fn_keyword,
    for_keyword,
    if_keyword,
    or_keyword,
    return_keyword,
    struct_keyword,
    use_keyword,
    while_keyword,
    true_keyword,
    false_keyword,
};

const Token = struct {
    info: TokenInfo,
    source_line: u32,

    pub fn deinit(self: Token, allocator: *Allocator) void {
        switch (self.info) {
            .string_literal => allocator.free(self.info.string_literal.text),
            .identifier => allocator.free(self.info.identifier.text),
            else => {},
        }
    }

    pub fn format(
        self: Token,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        switch (self.info) {
            .identifier => try writer.print("IDENT({s})", .{self.info.identifier.text}),
            .string_literal => try writer.writeAll("TODO_STR"),
            .int_literal => try writer.print("INT({})", .{self.info.int_literal.value}),
            .float_literal => try writer.print("FLOAT({})", .{self.info.float_literal.value}),
            else => {
                for (@tagName(self.info)) |char| {
                    try writer.print("{c}", .{std.ascii.toUpper(char)});
                }
            },
        }
    }
};

pub fn tokenize(allocator: *Allocator, text: []const u8) !ArrayList(Token) {
    var remaining_text = text[0..];
    var tokens = ArrayList(Token).init(allocator);
    var line_number: u32 = 1;

    skip_whitespace(&remaining_text);
    while (remaining_text.len != 0) {
        var char = remaining_text[0];
        skip_byte(&remaining_text);
        switch (char) {
            '\n' => line_number += 1,
            ' ', '\t', '\r' => {},
            '.' => try tokens.append(Token{ .info = .dot, .source_line = line_number }),
            ',' => try tokens.append(Token{ .info = .comma, .source_line = line_number }),
            '(' => try tokens.append(Token{ .info = .left_paren, .source_line = line_number }),
            ')' => try tokens.append(Token{ .info = .right_paren, .source_line = line_number }),
            '{' => try tokens.append(Token{ .info = .left_curly, .source_line = line_number }),
            '}' => try tokens.append(Token{ .info = .right_curly, .source_line = line_number }),
            '[' => try tokens.append(Token{ .info = .left_square, .source_line = line_number }),
            ']' => try tokens.append(Token{ .info = .right_square, .source_line = line_number }),
            '+' => try tokens.append(Token{ .info = .plus, .source_line = line_number }),
            ';' => try tokens.append(Token{ .info = .semicolon, .source_line = line_number }),
            ':' => try tokens.append(Token{ .info = if (match(&remaining_text, ':')) .double_colon else .colon, .source_line = line_number }),
            '<' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .lte else .left_angle, .source_line = line_number }),
            '>' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .gte else .right_angle, .source_line = line_number }),
            '-' => try tokens.append(Token{ .info = if (match(&remaining_text, '>')) .arrow else .minus, .source_line = line_number }),
            '*' => try tokens.append(Token{ .info = if (match(&remaining_text, '*')) .double_star else .star, .source_line = line_number }),
            '=' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .double_equal else .equal, .source_line = line_number }),
            '!' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .bang_equal else .bang, .source_line = line_number }),
            '/' => {
                if (match(&remaining_text, '/')) {
                    while (remaining_text.len >= 1 and remaining_text[0] != '\n') {
                        skip_byte(&remaining_text);
                    }
                } else {
                    try tokens.append(Token{ .info = .slash, .source_line = line_number });
                }
            },
            '"' => {
                // TODO: interpolated strings
                // TODO: other characters for string literals
                // TODO: escape sequences
                while (remaining_text.len >= 1 and remaining_text[0] != '"') {
                    skip_byte(&remaining_text);
                }
                skip_byte(&remaining_text);
                // TODO: actually copy out the text
                // For now we allocate a size-zero pointer so that we have
                // something to free during deinit
                try tokens.append(Token{ .info = .{ .string_literal = .{ .text = try allocator.alloc(u8, 0) } }, .source_line = 24601 });
            },
            '0'...'9' => {
                var number_span = find_number_end_pos(remaining_text);

                var number = remaining_text[0..number_span.end_pos];
                number.ptr -= 1;
                number.len += 1;

                switch (number_span.number_type) {
                    .int => try tokens.append(Token{ .info = .{ .int_literal = .{ .value = try std.fmt.parseInt(i128, number, 10) } }, .source_line = line_number }),
                    .float => try tokens.append(Token{ .info = .{ .float_literal = .{ .value = try std.fmt.parseFloat(f128, number) } }, .source_line = line_number }),
                }
                remaining_text = remaining_text[number_span.end_pos..];
            },
            'A'...'Z', 'a'...'z', '_' => {
                var end_pos = find_identifier_end_pos(remaining_text);

                // Do a bit of pointer twiddling to get the entire identifier,
                // even though remaining_text is already past the first byte.
                var ident = remaining_text[0..end_pos];
                ident.ptr -= 1;
                ident.len += 1;

                if (keyword_to_token(ident, line_number)) |token| {
                    try tokens.append(token);
                } else {
                    const ident_copy = try allocator.alloc(u8, end_pos + 1);
                    std.mem.copy(u8, ident_copy, ident);
                    try tokens.append(Token{
                        .info = .{ .identifier = .{ .text = ident_copy } },
                        .source_line = line_number,
                    });
                }
                remaining_text = remaining_text[end_pos..];
            },
            else => std.log.err("Don't know how to tokenize: {c}", .{char}),
        }
        skip_whitespace(&remaining_text);
    }

    return tokens;
}

pub fn keyword_to_token(text: []const u8, line_number: u32) ?Token {
    const keyword_table = .{
        .@"and" = TokenTag.and_keyword,
        .@"as" = TokenTag.as_keyword,
        .@"break" = TokenTag.break_keyword,
        .@"continue" = TokenTag.continue_keyword,
        .@"elif" = TokenTag.elif_keyword,
        .@"else" = TokenTag.else_keyword,
        .@"enum" = TokenTag.enum_keyword,
        .@"fn" = TokenTag.fn_keyword,
        .@"for" = TokenTag.for_keyword,
        .@"if" = TokenTag.if_keyword,
        .@"or" = TokenTag.or_keyword,
        .@"return" = TokenTag.return_keyword,
        .@"struct" = TokenTag.struct_keyword,
        .@"use" = TokenTag.use_keyword,
        .@"while" = TokenTag.while_keyword,
        .@"true" = TokenTag.true_keyword,
        .@"false" = TokenTag.false_keyword,
    };

    // TODO: comptime trie for finding keywords :)
    inline for (std.meta.fields(@TypeOf(keyword_table))) |field| {
        if (std.mem.eql(u8, text, field.name)) {
            return Token{
                .info = @field(keyword_table, field.name),
                .source_line = line_number,
            };
        }
    }

    return null;
}

pub fn match(text: *[]const u8, char: usize) bool {
    if (text.*.len >= 1 and text.*[0] == char) {
        skip_byte(text);
        return true;
    }
    return false;
}

pub fn find_identifier_end_pos(text: []const u8) u32 {
    var pos: u32 = 0;
    while (pos < text.len) {
        switch (text[pos]) {
            'A'...'Z', 'a'...'z', '_', '0'...'9' => {},
            else => return pos,
        }
        pos += 1;
    }
    return pos;
}

const NumberSpan = struct {
    number_type: enum { int, float },
    end_pos: u32,
};

// TODO: hex and binary literals
// TODO: scientific notation
pub fn find_number_end_pos(text: []const u8) NumberSpan {
    var isFloat = false;
    var pos: u32 = 0;
    while (pos < text.len) {
        switch (text[pos]) {
            '0'...'9' => {},
            '.' => {
                if (isFloat) {
                    return NumberSpan{ .number_type = .float, .end_pos = pos };
                }
                isFloat = true;
            },
            else => break,
        }
        pos += 1;
    }

    if (isFloat) {
        return NumberSpan{ .number_type = .float, .end_pos = pos };
    } else {
        return NumberSpan{ .number_type = .int, .end_pos = pos };
    }
}

pub fn parse_text(allocator: *Allocator, text: []const u8) !Ast {
    var stmts = ArrayList(Statement).init(allocator);
    errdefer stmts.deinit();

    var tokens = try tokenize(allocator, text);
    defer {
        for (tokens.items) |token| {
            token.deinit(allocator);
        }
        tokens.deinit();
    }

    var remaining_tokens = tokens.items;

    while (have_tokens(&remaining_tokens)) {
        var stmt = try parse_statement(allocator, &remaining_tokens);
        try stmts.append(stmt);
    }

    return Ast{ .allocator = allocator, .stmts = stmts };
}

pub fn skip_whitespace(text: *[]const u8) void {
    while (have_text(text)) {
        switch (text.*[0]) {
            ' ' => skip_byte(text),
            '\n' => skip_byte(text),
            '\t' => skip_byte(text),
            '\r' => skip_byte(text),
            '/' => if (text.*.len >= 2 and text.*[1] == '/') {
                skip_bytes(text, 2);
                // Consume all characters until we hit a newline
                while (have_text(text)) {
                    switch (text.*[0]) {
                        '\n' => {
                            skip_byte(text);
                            break;
                        },
                        else => skip_byte(text),
                    }
                }
            } else {
                return;
            },
            else => return,
        }
    }
}

pub fn skip_bytes(text: *[]const u8, count: usize) void {
    text.* = text.*[count..];
}

pub fn skip_byte(text: *[]const u8) void {
    skip_bytes(text, 1);
}

pub fn skip_all(comptime T: type, slice: *[]const T) void {
    slice.* = slice.*[slice.*.len..];
}

pub fn parse_statement(allocator: *Allocator, tokens: *[]const Token) !Statement {
    _ = allocator;

    if (!have_tokens(tokens)) {
        return error.RanOutOfTokens;
    }

    if (try parse_use_statement(allocator, tokens)) |stmt| {
        return stmt;
    } else if (try parse_expressionlike_statement(allocator, tokens)) |stmt| {
        return stmt;
    }

    return error.ParseError;
}

pub fn parse_use_statement(allocator: *Allocator, tokens: *[]const Token) !?Statement {
    _ = allocator;
    _ = tokens;

    if (!have_tokens(tokens)) {
        return error.RanOutOfTokens;
    }

    switch (tokens.*[0].info) {
        .use_keyword => {
            // TODO: actually parse this
            skip_all(Token, tokens);
            return error.NoParsingUseStatements;
        },
        else => return null,
    }
}

pub fn parse_expressionlike_statement(allocator: *Allocator, tokens: *[]const Token) !?Statement {
    _ = allocator;
    _ = tokens;

    var slice_copy = tokens.*;
    if (try parse_expression(allocator, &slice_copy)) |expr| {
        // Oh no... this is going to need to go into a lot of places...
        errdefer expr.deinit(allocator);

        switch (try peek_token(&slice_copy)) {
            .equal => {
                // TODO: turn the expression we got into an assignment
                skip_all(Token, tokens);
                return error.AssignmentNotImplemented;
            },
            .semicolon => {
                try consume_token(&slice_copy);
                // Only advance the token slice when we're successful
                tokens.* = slice_copy;
                return Statement{ .expression_statement = expr };
            },
            else => {
                std.log.info("Next token is: {}", .{try peek_token(&slice_copy)});
                return null;
            },
        }
    }

    std.log.info("Not an expression statement 1", .{});
    return null;
}

pub fn peek_tokenx(tokens: *[]const Token, idx: usize) !TokenInfo {
    if (tokens.*.len > idx) {
        return tokens.*[idx].info;
    }
    return error.RanOutOfTokens;
}

pub fn peek_token(tokens: *[]const Token) !TokenInfo {
    return peek_tokenx(tokens, 0);
}

pub fn consume_token(tokens: *[]const Token) !void {
    if (have_tokens(tokens)) {
        tokens.* = tokens.*[1..];
        return;
    }
    return error.RanOutOfTokens;
}

pub fn parse_expression(allocator: *Allocator, tokens: *[]const Token) !?Expression {
    if (try parse_logic_or(allocator, tokens)) |expr| {
        return expr;
    }

    return null;
}

pub fn parse_precedence(
    comptime higher_precedence: fn (*Allocator, *[]const Token) anyerror!?Expression,
    comptime is_logical: bool,
    comptime op_table: anytype,
) fn (*Allocator, *[]const Token) anyerror!?Expression {
    return struct {
        fn parser(allocator: *Allocator, tokens: *[]const Token) anyerror!?Expression {
            var slice_copy = tokens.*;
            if (try higher_precedence(allocator, &slice_copy)) |higher_precedence_expr1| {
                var expr: Expression = higher_precedence_expr1;

                outer: while (have_tokens(&slice_copy)) {
                    inline for (std.meta.fields(@TypeOf(op_table))) |field| {
                        var token_name: []const u8 = @tagName(try peek_token(&slice_copy));
                        var field_name: []const u8 = field.name;

                        var should_return = false;
                        if (std.mem.eql(u8, token_name, field_name)) {
                            try consume_token(&slice_copy);
                            var maybe_expr: ?Expression = try higher_precedence(allocator, &slice_copy);
                            if (maybe_expr) |higher_precedence_expr| {
                                var left: *Expression = try allocator.create(Expression);
                                var right: *Expression = try allocator.create(Expression);
                                left.* = expr;
                                right.* = higher_precedence_expr;

                                if (is_logical) {
                                    expr = Expression{ .logical = LogicalExpr{ .left = left, .op = @field(op_table, field.name), .right = right } };
                                } else {
                                    expr = Expression{ .binary = BinaryExpr{ .left = left, .op = @field(op_table, field.name), .right = right } };
                                }

                                // Return to the top to continue along chained
                                // binary operations at this same precedence.
                                continue :outer;
                            } else {
                                // Don't return here. Provide indirection to get around https://github.com/ziglang/zig/issues/8893
                                should_return = true;
                            }

                            if (should_return) {
                                return null;
                            }
                        }
                    }

                    // Didn't match any recognized tokens
                    break;
                }

                tokens.* = slice_copy;
                return expr;
            }

            return null;
        }
    }.parser;
}

const parse_logic_or = parse_precedence(
    parse_logic_and,
    true,
    .{ .or_keyword = .LogicalOr },
);

const parse_logic_and = parse_precedence(
    parse_equality,
    true,
    .{ .and_keyword = .LogicalAnd },
);

const parse_equality = parse_precedence(
    parse_comparison,
    false,
    .{
        .bang_equal = .BinaryBangEqual,
        .double_equal = .BinaryDoubleEqual,
    },
);

const parse_comparison = parse_precedence(
    parse_term,
    false,
    .{
        .gte = .BinaryGreaterThanEqual,
        .lte = .BinaryLessThanEqual,
        .right_angle = .BinaryGreaterThan,
        .left_angle = .BinaryLessThan,
    },
);

const parse_term = parse_precedence(
    parse_factor,
    false,
    .{
        .plus = .BinaryAdd,
        .minus = .BinarySub,
    },
);

const parse_factor = parse_precedence(
    parse_unary,
    false,
    .{
        .star = .BinaryMul,
        .slash = .BinaryDiv,
    },
);

pub fn parse_unary(allocator: *Allocator, tokens: *[]const Token) anyerror!?Expression {
    if (have_tokens(tokens)) {
        switch (try peek_token(tokens)) {
            .bang => {
                var slice_copy = tokens.*;
                try consume_token(&slice_copy);

                if (try parse_unary(allocator, &slice_copy)) |expr| {
                    var left = try allocator.create(Expression);
                    left.* = expr;

                    tokens.* = slice_copy;
                    return Expression{ .unary = .{ .expr = left, .op = .UnaryNot } };
                }
                return null;
            },
            .minus => {
                var slice_copy = tokens.*;
                try consume_token(&slice_copy);

                if (try parse_unary(allocator, &slice_copy)) |expr| {
                    var left = try allocator.create(Expression);
                    left.* = expr;

                    tokens.* = slice_copy;
                    return Expression{ .unary = .{ .expr = left, .op = .UnaryNegation } };
                }
                return null;
            },
            else => return try parse_call(allocator, tokens),
        }
    }

    return null;
}

pub fn parse_call(allocator: *Allocator, tokens: *[]const Token) !?Expression {
    var slice_copy = tokens.*;
    if (try parse_primary(allocator, &slice_copy)) |got_expr| {
        var expr = got_expr;
        errdefer expr.deinit(allocator);

        while (have_tokens(&slice_copy)) {
            switch (try peek_token(&slice_copy)) {
                .left_paren => {
                    try consume_token(&slice_copy);
                    var maybe_arg: ?Expression = null;
                    // TODO: support multiple arguments
                    if (try parse_expression(allocator, &slice_copy)) |arg| {
                        maybe_arg = arg;
                    }

                    switch (try peek_token(&slice_copy)) {
                        .right_paren => {
                            try consume_token(&slice_copy);
                            var left = try allocator.create(Expression);
                            left.* = expr;

                            if (maybe_arg) |arg| {
                                var args = try allocator.create(Expression);
                                args.* = arg;

                                expr = Expression{ .call = .{ .left = left, .args = args } };
                            } else {
                                expr = Expression{ .call = .{ .left = left, .args = null } };
                            }
                        },
                        else => {
                            // Expected a closing paren.
                            // TODO: there _has_ to be a better way to do this...
                            expr.deinit(allocator);
                            return null;
                        },
                    }
                },
                else => break,
            }
        }

        tokens.* = slice_copy;
        return expr;
    }

    return null;
}

pub fn parse_primary(allocator: *Allocator, tokens: *[]const Token) !?Expression {
    if (have_tokens(tokens)) {
        switch (try peek_token(tokens)) {
            .true_keyword => {
                try consume_token(tokens);
                return Expression{ .bool_literal = true };
            },
            .false_keyword => {
                try consume_token(tokens);
                return Expression{ .bool_literal = false };
            },
            .int_literal => |int_literal| {
                try consume_token(tokens);
                return Expression{ .int_literal = int_literal.value };
            },
            .string_literal => |string_literal| {
                try consume_token(tokens);
                return Expression{ .string_literal = string_literal.text };
            },
            .identifier => |token_ident| {
                try consume_token(tokens);
                var ident = try allocator.alloc(u8, token_ident.text.len);
                std.mem.copy(u8, ident, token_ident.text);
                return Expression{ .identifier = ident };
            },
            .left_paren => {
                var slice_copy = tokens.*;
                try consume_token(&slice_copy);
                if (have_tokens(&slice_copy)) {
                    if (try parse_expression(allocator, &slice_copy)) |expr| {
                        switch (try peek_token(&slice_copy)) {
                            .right_paren => {
                                try consume_token(&slice_copy);
                                tokens.* = slice_copy;
                                return expr;
                            },
                            else => {},
                        }
                    }
                }
            },
            else => {},
        }
    }

    return null;
}

// primary        → "true" | "false"
//                | NUMBER | STRING | IDENTIFIER | "(" expression ")";

pub fn have_text(text: *[]const u8) bool {
    return text.*.len >= 1;
}

pub fn have_tokens(tokens: *[]const Token) bool {
    return tokens.*.len >= 1;
}

pub fn interpret_ast(ast: Ast) void {
    for (ast.stmts.items) |stmt| {
        _ = interpret_stmt(stmt);
    }
}

pub fn interpret_stmt(stmt: Statement) ShimValue {
    switch (stmt) {
        StatementTag.expression_statement => {
            var result = interpret_expr(&stmt.expression_statement);
            return result;
        },
        // TODO: lies
        StatementTag.if_statement => unreachable,
        StatementTag.pretend_statement => return ShimValue{ .shim_unit = ShimUnit{} },
        // TODO: lies
        StatementTag.assignment_statement => unreachable,
    }
    return ShimValue{ .shim_unit = ShimUnit{} };
}

pub fn interpret_expr(expr: *const Expression) ShimValue {
    switch (expr.*) {
        ExpressionTag.int_literal => |v| return ShimValue{ .shim_i128 = v },
        ExpressionTag.string_literal => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.bool_literal => |b| return ShimValue{ .shim_bool = b },
        ExpressionTag.unary => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.logical => |_| unreachable,
        ExpressionTag.binary => |bexpr| {
            var left = switch (interpret_expr(bexpr.left)) {
                ValueTag.shim_i128 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            var right = switch (interpret_expr(bexpr.right)) {
                ValueTag.shim_i128 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            switch (bexpr.op) {
                BinaryOperator.BinaryAdd => return ShimValue{ .shim_i128 = left + right },
                BinaryOperator.BinaryMul => return ShimValue{ .shim_i128 = left * right },
                else => unreachable,
            }
        },
        ExpressionTag.identifier => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.call => |cexpr| {
            // Ignore the thing being called... we always assume it's print for now :)
            if (cexpr.args) |args| {
                var val = interpret_expr(args);
                switch (val) {
                    .shim_i128 => |num| std.debug.print("{}", .{num}),
                    else => unreachable,
                }
            }
            std.debug.print("\n", .{});

            return ShimValue{ .shim_unit = ShimUnit{} };
        },
    }
}
