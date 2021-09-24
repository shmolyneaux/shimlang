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
const StatementTag = enum { expr, if_statement, pretend_statement };

const Statement = union(StatementTag) {
    expr: Expression,
    if_statement: IfStatement,
    // Ignore this for now, I just need a placeholder
    pretend_statement: bool,

    pub fn deinit(self: Statement, allocator: *Allocator) void {
        switch (self) {
            StatementTag.expr => self.expr.deinit(allocator),
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
            StatementTag.expr => try writer.print("STATEMENT({})", .{self.expr}),
            StatementTag.if_statement => try writer.print("STATEMENT({})", .{self.if_statement}),
            StatementTag.pretend_statement => try writer.print("STATEMENT(PRETEND())", .{}),
        }
    }
};

const UnaryOperator = enum {
    UnaryNegation,
    UnaryNot,
};
const UnaryExpr = struct {
    expr: u8,
    op: UnaryOperator,
};

const BinaryOperator = enum {
    BinaryAdd,
    BinaryMul,
};

const BinaryExpr = struct {
    left: *Expression,
    op: BinaryOperator,
    right: *Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: BinaryExpr, allocator: *Allocator) void {
        allocator.destroy(self.left);
        allocator.destroy(self.right);
    }
};

const ExpressionTag = enum {
    int_literal,
    string_literal,
    unary,
    binary,
    call,
};
const Expression = union(ExpressionTag) {
    int_literal: u8,
    string_literal: []const u8,
    unary: UnaryExpr,
    binary: BinaryExpr,
    call: bool,

    pub fn deinit(self: Expression, allocator: *Allocator) void {
        switch (self) {
            ExpressionTag.int_literal => |_| {},
            ExpressionTag.string_literal => |_| {},
            ExpressionTag.unary => |_| {},
            ExpressionTag.binary => |bexpr| bexpr.deinit(allocator),
            ExpressionTag.call => |_| {},
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

        try writer.print("EXPRESSION({s})", .{"foo"});
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
    shim_u8,
    // shim_u16,
    // shim_u32,
    // shim_u64,
    // shim_i8,
    // shim_i16,
    // shim_i32,
    // shim_i64,

    // shim_bool

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
    shim_u8: u8,
    shim_str: bool,
    shim_unit: ShimUnit,
    shim_error: ShimUnit,
    shim_obj: ShimUnit,
};

// Functions!

pub fn run_text(allocator: *Allocator, text: []const u8) !void {
    var ast = try parse_text(allocator, text);
    defer ast.deinit();

    std.debug.print("{any}\n", .{ast});
    var result = interpret_ast(ast);

    std.debug.print("{}\n", .{result});
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
    arrow,
    double_equal,
    equal,
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
};
const Token = struct {
    info: union(TokenTag) {
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
        arrow,
        double_equal,
        equal,
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
    },
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
            ':' => try tokens.append(Token{ .info = if (match(&remaining_text, ':')) .double_colon else .colon, .source_line = line_number }),
            '<' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .lte else .left_angle, .source_line = line_number }),
            '>' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .gte else .right_angle, .source_line = line_number }),
            '-' => try tokens.append(Token{ .info = if (match(&remaining_text, '>')) .arrow else .minus, .source_line = line_number }),
            '*' => try tokens.append(Token{ .info = if (match(&remaining_text, '*')) .double_star else .star, .source_line = line_number }),
            '=' => try tokens.append(Token{ .info = if (match(&remaining_text, '=')) .double_equal else .equal, .source_line = line_number }),
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

    std.log.info("Got tokens: {}", .{tokens});
    return tokens;
}

pub fn keyword_to_token(text: []const u8, line_number: u32) ?Token {
    const keyword_table = .{
        .@"and" = TokenTag.and_keyword,
        .@"as_keyword" = TokenTag.as_keyword,
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
        .@"use_keyword" = TokenTag.use_keyword,
        .@"while" = TokenTag.while_keyword,
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
    _ = stmts;

    var tokens = try tokenize(allocator, text);
    defer {
        for (tokens.items) |token| {
            token.deinit(allocator);
        }
        tokens.deinit();
    }

    var remaining_tokens = tokens.items;

    while (remaining_tokens.len != 0) {
        var stmt = try parse_statement(allocator, &remaining_tokens);
        try stmts.append(stmt);
    }

    return Ast{ .allocator = allocator, .stmts = stmts };
}

pub fn skip_whitespace(text: *[]const u8) void {
    while (text.*.len != 0) {
        switch (text.*[0]) {
            ' ' => skip_byte(text),
            '\n' => skip_byte(text),
            '\t' => skip_byte(text),
            '\r' => skip_byte(text),
            '/' => if (text.*.len >= 2 and text.*[1] == '/') {
                skip_bytes(text, 2);
                // Consume all characters until we hit a newline
                while (text.*.len != 0) {
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

pub fn parse_statement(allocator: *Allocator, tokens: *[]const Token) !Statement {
    _ = allocator;
    tokens.* = tokens.*[tokens.*.len..];
    return Statement{ .pretend_statement = true };
}

pub fn parse_expression(allocator: *Allocator, text: *[]const u8) !?Expression {
    _ = allocator;
    _ = text;

    // TODO: this is how we consume text functionally
    text.* = text.*[1..];
    std.log.info("blah: {s}", .{text.*});

    return null;
}

pub fn interpret_ast(ast: Ast) void {
    for (ast.stmts.items) |stmt| {
        _ = interpret_stmt(stmt);
    }
}

pub fn interpret_stmt(stmt: Statement) ShimValue {
    switch (stmt) {
        StatementTag.expr => return interpret_expr(&stmt.expr),
        StatementTag.if_statement => unreachable,
        StatementTag.pretend_statement => return ShimValue{ .shim_unit = ShimUnit{} },
    }
    return ShimValue{ .shim_unit = ShimUnit{} };
}

pub fn interpret_expr(expr: *const Expression) ShimValue {
    switch (expr.*) {
        ExpressionTag.int_literal => |v| return ShimValue{ .shim_u8 = v },
        ExpressionTag.string_literal => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.unary => |_| return ShimValue{ .shim_str = false },
        ExpressionTag.binary => |bexpr| {
            var left = switch (interpret_expr(bexpr.left)) {
                ValueTag.shim_u8 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            var right = switch (interpret_expr(bexpr.right)) {
                ValueTag.shim_u8 => |v| v,
                else => return ShimValue{ .shim_unit = ShimUnit{} },
            };
            switch (bexpr.op) {
                BinaryOperator.BinaryAdd => return ShimValue{ .shim_u8 = left + right },
                BinaryOperator.BinaryMul => return ShimValue{ .shim_u8 = left * right },
            }
        },
        ExpressionTag.call => |_| return ShimValue{ .shim_str = false },
    }
}

// TODO: this is how we could conditionally add comments to the AST
// const build_options = @import("build_options");
//
// const Ast = blk: {
//     if (build_options.include_comments_in_ast) {
//         break :blk struct {
//             foo: u8,
//             comment: u8,
//         };
//     } else {
//         break :blk struct {
//             foo: u8,
//         };
//     }
// };
