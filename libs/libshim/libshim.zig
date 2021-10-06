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
    block: Block,

    pub fn deinit(self: Ast) void {
        self.block.deinit();
    }

    pub fn format(
        self: Ast,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("AST({})", self.block);
    }
};

const Block = struct {
    allocator: *Allocator,
    stmts: ArrayList(Statement),

    pub fn deinit(self: Block) void {
        for (self.stmts.items) |stmt| {
            stmt.deinit(self.allocator);
        }
        self.stmts.deinit();
    }

    pub fn format(
        self: Block,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.writeAll("[");

        for (self.stmts.items) |stmt, i| {
            try writer.print("{}", .{stmt});
            if (i + 1 != self.stmts.len) {
                try writer.writeAll(",");
            }
        }

        try writer.writeAll("]");
    }
};

const IfStatement = struct { predicate: Expression, if_block: Block, else_block: ?Block };

// TODO: Multiple sorts of statements (if, for, use, while, etc.)
const StatementTag = enum { expression_statement, if_statement, declaration_statement, assignment_statement, break_statement, while_statement };

const Statement = union(StatementTag) {
    expression_statement: Expression,
    if_statement: IfStatement,
    break_statement,
    while_statement: struct { predicate: Expression, block: Block },
    declaration_statement: struct { name: []const u8, value: Expression },
    assignment_statement: struct { obj: ?Expression, name: []const u8, value: Expression },

    pub fn deinit(self: Statement, allocator: *Allocator) void {
        switch (self) {
            StatementTag.expression_statement => |stmt| stmt.deinit(allocator),
            StatementTag.if_statement => |stmt| {
                stmt.predicate.deinit(allocator);
                stmt.if_block.deinit();
                if (stmt.else_block) |else_block| {
                    else_block.deinit();
                }
            },
            StatementTag.break_statement => return,
            StatementTag.while_statement => |stmt| {
                stmt.predicate.deinit(allocator);
                stmt.block.deinit();
            },
            StatementTag.declaration_statement => |stmt| {
                allocator.free(stmt.name);
                stmt.value.deinit(allocator);
            },
            StatementTag.assignment_statement => |stmt| {
                allocator.free(stmt.name);
                stmt.value.deinit(allocator);
                if (stmt.obj) |obj| {
                    obj.deinit(allocator);
                }
            },
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
    args: []const Expression,

    // TODO: We assume that some parent structure provides the allocator that
    // was used to originally create these expressions.
    pub fn deinit(self: CallExpr, allocator: *Allocator) void {
        self.left.deinit(allocator);
        allocator.destroy(self.left);
        for (self.args) |arg| {
            arg.deinit(allocator);
        }
        allocator.free(self.args);
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
    attribute,
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
    attribute: struct { left: *Expression, name: []const u8 },
    call: CallExpr,

    pub fn deinit(self: Expression, allocator: *Allocator) void {
        switch (self) {
            ExpressionTag.int_literal => |_| {},
            ExpressionTag.string_literal => |str| allocator.free(str),
            ExpressionTag.bool_literal => |_| {},
            ExpressionTag.unary => |_| {},
            ExpressionTag.identifier => |ident| allocator.free(ident),
            ExpressionTag.logical => |lexpr| lexpr.deinit(allocator),
            ExpressionTag.binary => |bexpr| bexpr.deinit(allocator),
            ExpressionTag.attribute => |aexpr| {
                aexpr.left.deinit(allocator);
                allocator.destroy(aexpr.left);
                allocator.free(aexpr.name);
            },
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
                for (val.args) |arg, i| {
                    try writer.print("{}", .{arg});
                    if (i != val.args.len - 1) {
                        try writer.print(",", .{});
                    }
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
    shim_bound_method,
    shim_native_fn,
};
const ShimValue = union(ValueTag) {
    shim_i128: i128,
    shim_str: []const u8,
    shim_bool: bool,
    shim_unit: ShimUnit,
    shim_error: ShimUnit,
    shim_obj: ShimUnit,
    shim_bound_method: struct { obj: *ShimValue, func: *ShimValue },
    // In reality this is a pointer to something that returns a ShimValue, but zig doesn't like that
    shim_native_fn: PretendNativeFn,

    const RealNativeFn = *const fn (*Allocator, []const ShimValue) anyerror!ShimValue;
    const PretendNativeFn = *const fn (*Allocator, []const ShimValue) anyerror!void;

    fn get(self: ShimValue, allocator: *Allocator, name: []const u8) anyerror!ShimValue {
        switch (self) {
            .shim_str => {
                if (std.mem.eql(u8, name, "lower")) {
                    var obj_ref = try allocator.create(ShimValue);
                    errdefer allocator.destroy(obj_ref);

                    obj_ref.* = try self.clone(allocator);
                    errdefer obj_ref.deinit(allocator);

                    var func_ref = try allocator.create(ShimValue);
                    func_ref.* = create_native_fn(&native_str_lower);

                    return ShimValue{ .shim_bound_method = .{ .obj = obj_ref, .func = func_ref } };
                }

                return error.StrOnlyHasLower;
            },
            else => return error.GetNotImplementedOnValue,
        }
    }

    fn call(func: ShimValue, allocator: *Allocator, args: []const ShimValue) anyerror!ShimValue {
        switch (func) {
            .shim_bound_method => |meth| {
                var args_with_obj = try allocator.alloc(ShimValue, args.len + 1);
                args_with_obj[0] = meth.obj.*;
                defer allocator.free(args_with_obj);
                std.mem.copy(ShimValue, args_with_obj[1..], args);

                return try meth.func.*.call(allocator, args_with_obj);
            },
            .shim_native_fn => return func.get_native_fn().*(allocator, args),
            else => return error.ValueCannotBeCalled,
        }
    }

    fn is_truthy(self: ShimValue) bool {
        return switch (self) {
            .shim_i128 => |val| val != 0,
            .shim_str => |val| val.len != 0,
            .shim_bool => |val| val,
            .shim_unit => false,
            .shim_error => false,
            .shim_native_fn => true,
            else => return true, // Return true for now, call __bool__ later
        };
    }

    fn create_native_fn(func: RealNativeFn) ShimValue {
        return ShimValue{ .shim_native_fn = @ptrCast(PretendNativeFn, func) };
    }

    fn get_native_fn(self: ShimValue) RealNativeFn {
        return @ptrCast(RealNativeFn, self.shim_native_fn);
    }

    fn clone(self: ShimValue, allocator: *Allocator) !ShimValue {
        switch (self) {
            .shim_i128 => return self,
            .shim_str => |str| return ShimValue{ .shim_str = try copy_slice(allocator, str) },
            .shim_bool => return self,
            .shim_unit => return self,
            .shim_error => return self,
            .shim_obj => return self,
            .shim_bound_method => |meth| {
                var obj_ref = try allocator.create(ShimValue);
                var func_ref = try allocator.create(ShimValue);
                obj_ref.* = meth.obj.*;
                func_ref.* = meth.func.*;

                return ShimValue{ .shim_bound_method = .{ .obj = obj_ref, .func = func_ref } };
            },
            .shim_native_fn => return self,
        }
    }

    fn deinit(self: ShimValue, allocator: *Allocator) void {
        switch (self) {
            .shim_str => |str| allocator.free(str),
            .shim_bound_method => |meth| {
                meth.obj.*.deinit(allocator);
                meth.func.*.deinit(allocator);
                allocator.destroy(meth.obj);
                allocator.destroy(meth.func);
            },
            else => {},
        }
    }
};

// Functions!

pub fn run_text(allocator: *Allocator, text: []const u8) !void {
    var ast = try parse_text(allocator, text);
    defer ast.deinit();

    var interpreter = Interpreter.init(allocator);
    defer interpreter.deinit();

    try interpreter.interpret_ast(ast);
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
    else_keyword,
    enum_keyword,
    fn_keyword,
    for_keyword,
    if_keyword,
    or_keyword,
    return_keyword,
    struct_keyword,
    use_keyword,
    let_keyword,
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
    else_keyword,
    enum_keyword,
    fn_keyword,
    for_keyword,
    if_keyword,
    or_keyword,
    return_keyword,
    struct_keyword,
    use_keyword,
    let_keyword,
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
                var byte_count: u32 = 0;
                var pos: u32 = 0;
                var found_closing_quote = false;
                while (pos < remaining_text.len) {
                    switch (remaining_text[pos]) {
                        '"' => {
                            found_closing_quote = true;
                            break;
                        },
                        '\\' => {
                            // Ignore the character after the backslash
                            pos += 1;
                        },
                        else => {},
                    }
                    byte_count += 1;
                    pos += 1;
                }
                if (!found_closing_quote) {
                    return error.StringNotClosed;
                }
                var end_pos = pos;

                // TODO: interpolated strings
                // TODO: other characters for string literals (single quote)

                const str_copy = try allocator.alloc(u8, byte_count);
                var dest_idx: u32 = 0;
                var escape_next = false;

                {
                    // Is this actually how you do a c-style for loop?
                    var src_idx: u32 = 0;
                    while (src_idx < end_pos) {
                        defer src_idx += 1;

                        str_copy[dest_idx] = blk: {
                            if (escape_next) {
                                escape_next = false;
                                break :blk switch (remaining_text[src_idx]) {
                                    'n' => '\n',
                                    't' => '\t',
                                    'r' => '\r',
                                    // Use the exact byte that appears after the slash
                                    else => remaining_text[src_idx],
                                };
                            } else {
                                if (remaining_text[src_idx] == '\\') {
                                    escape_next = true;
                                    continue;
                                }
                                break :blk remaining_text[src_idx];
                            }
                        };
                        dest_idx += 1;
                    }
                }

                // We add 1 to the end position to consume the closing '"'
                remaining_text = remaining_text[end_pos + 1 ..];

                try tokens.append(Token{ .info = .{ .string_literal = .{ .text = str_copy } }, .source_line = 24601 });
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
                    const ident_copy = try copy_slice(allocator, ident);
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
        .@"else" = TokenTag.else_keyword,
        .@"enum" = TokenTag.enum_keyword,
        .@"fn" = TokenTag.fn_keyword,
        .@"for" = TokenTag.for_keyword,
        .@"if" = TokenTag.if_keyword,
        .@"or" = TokenTag.or_keyword,
        .@"return" = TokenTag.return_keyword,
        .@"struct" = TokenTag.struct_keyword,
        .@"use" = TokenTag.use_keyword,
        .@"let" = TokenTag.let_keyword,
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
    errdefer {
        for (stmts.items) |stmt| {
            stmt.deinit(allocator);
        }
        stmts.deinit();
    }

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
        errdefer stmt.deinit(allocator);

        try stmts.append(stmt);
    }

    return Ast{ .allocator = allocator, .block = .{ .allocator = allocator, .stmts = stmts } };
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

pub fn parse_statement(allocator: *Allocator, tokens: *[]const Token) anyerror!Statement {
    _ = allocator;

    if (!have_tokens(tokens)) {
        return error.RanOutOfTokens;
    }

    switch (try peek_token(tokens)) {
        .use_keyword => if (try parse_use_statement(allocator, tokens)) |stmt| {
            return stmt;
        },
        .let_keyword => return try parse_declaration(allocator, tokens),
        .if_keyword => return try parse_conditional_statement(allocator, tokens),
        .break_keyword => {
            try expect_token(tokens, .break_keyword);
            try expect_token(tokens, .semicolon);

            return Statement.break_statement;
        },
        .while_keyword => return try parse_while_statement(allocator, tokens),
        else => if (try parse_expressionlike_statement(allocator, tokens)) |stmt| {
            return stmt;
        },
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

pub fn parse_conditional_statement(allocator: *Allocator, tokens: *[]const Token) anyerror!Statement {
    _ = allocator;
    try expect_token(tokens, .if_keyword);

    var expr = blk: {
        if (try parse_expression(allocator, tokens)) |expr| {
            break :blk expr;
        } else {
            return error.ParseError;
        }
    };
    errdefer expr.deinit(allocator);

    var if_block = try parse_block(allocator, tokens);
    errdefer if_block.deinit();

    var else_block: ?Block = null;
    if (have_tokens(tokens)) {
        if (try match_token(tokens, .else_keyword)) |_| {
            if (try is_next_token(tokens, .if_keyword)) {
                var nested_if = try parse_conditional_statement(allocator, tokens);
                errdefer nested_if.deinit(allocator);

                var lst = ArrayList(Statement).init(allocator);
                errdefer lst.deinit();

                try lst.append(nested_if);

                else_block = Block{ .allocator = allocator, .stmts = lst };
            } else {
                var found_else_block = try parse_block(allocator, tokens);
                else_block = found_else_block;
            }
        }
    }

    return Statement{ .if_statement = .{
        .predicate = expr,
        .if_block = if_block,
        .else_block = else_block,
    } };
}

pub fn parse_while_statement(allocator: *Allocator, tokens: *[]const Token) anyerror!Statement {
    _ = allocator;
    try expect_token(tokens, .while_keyword);

    var expr = blk: {
        if (try parse_expression(allocator, tokens)) |expr| {
            break :blk expr;
        } else {
            return error.ParseError;
        }
    };
    errdefer expr.deinit(allocator);

    var block = try parse_block(allocator, tokens);

    return Statement{ .while_statement = .{
        .predicate = expr,
        .block = block,
    } };
}

pub fn parse_block(allocator: *Allocator, tokens: *[]const Token) !Block {
    try expect_token(tokens, .left_curly);

    var stmts = ArrayList(Statement).init(allocator);
    errdefer {
        for (stmts.items) |stmt| {
            stmt.deinit(allocator);
        }
        stmts.deinit();
    }

    while (!(try is_next_token(tokens, .right_curly))) {
        var stmt = try parse_statement(allocator, tokens);
        try stmts.append(stmt);
    }
    try expect_token(tokens, .right_curly);

    return Block{ .allocator = allocator, .stmts = stmts };
}

pub fn parse_declaration(allocator: *Allocator, tokens: *[]const Token) !Statement {
    _ = allocator;
    try expect_token(tokens, .let_keyword);
    if (try match_token(tokens, .identifier)) |token| {
        var name = try copy_slice(allocator, token.identifier.text);
        errdefer allocator.free(name);

        try expect_token(tokens, .equal);

        var maybe_expr = try parse_expression(allocator, tokens);
        if (maybe_expr) |expr| {
            try expect_token(tokens, .semicolon);
            return Statement{ .declaration_statement = .{ .name = name, .value = expr } };
        }

        std.debug.print("didnt get an expr\n", .{});
        return error.CantParseDeclaration;
    }
    return error.CantParseDeclaration;
}

pub fn parse_expressionlike_statement(allocator: *Allocator, tokens: *[]const Token) !?Statement {
    _ = allocator;
    _ = tokens;

    var slice_copy = tokens.*;
    if (try parse_expression(allocator, &slice_copy)) |expr| {
        // Oh no... this is going to need to go into a lot of places...
        errdefer expr.deinit(allocator);

        var token_type = peek_token(&slice_copy) catch |err| {
            std.debug.print("Expected '=' or ';' after expression\n", .{});
            return err;
        };
        switch (token_type) {
            .equal => {
                try consume_token(&slice_copy);

                switch (expr) {
                    .identifier => |name| {
                        if (try parse_expression(allocator, &slice_copy)) |value_expr| {
                            errdefer value_expr.deinit(allocator);

                            try expect_token(&slice_copy, .semicolon);

                            tokens.* = slice_copy;
                            return Statement{ .assignment_statement = .{ .obj = null, .name = name, .value = value_expr } };
                        }
                    },
                    else => return error.CanOnlyAssignToIdentifier,
                }

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
            else => {},
        }

        std.log.info("Next token is: {}", .{try peek_token(&slice_copy)});
        expr.deinit(allocator);
    }

    std.log.info("Not an expression statement 1: {s}", .{tokens.*});
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

/// Consume a token
pub fn consume_token(tokens: *[]const Token) !void {
    if (have_tokens(tokens)) {
        tokens.* = tokens.*[1..];
        return;
    }
    return error.RanOutOfTokens;
}

/// Check if the next token has a specific TokenTag, without consuming it
pub fn is_next_token(tokens: *[]const Token, expected: TokenTag) !bool {
    var token = try peek_token(tokens);
    if (@as(TokenTag, token) == expected) {
        return true;
    }

    return false;
}

/// Consume and return a token if it matches a tag, return null otherwise
// TODO: could this be comptime and return the actual type from the union?
pub fn match_token(tokens: *[]const Token, expected: TokenTag) !?TokenInfo {
    if (try is_next_token(tokens, expected)) {
        var token = try peek_token(tokens);
        tokens.* = tokens.*[1..];
        return token;
    }

    return null;
}

/// Consume a token if it matches a tag, error otherwise
pub fn expect_token(tokens: *[]const Token, expected: TokenTag) !void {
    _ = try expect_match_token(tokens, expected);
}

/// Consume and return a token if it matches a tag, error otherwise
pub fn expect_match_token(tokens: *[]const Token, expected: TokenTag) !TokenInfo {
    if (try match_token(tokens, expected)) |info| {
        return info;
    }

    var token_tag = try peek_token(tokens);
    std.debug.print("Expected {} but got {}\n", .{ expected, token_tag });
    return error.UnexpectedToken;
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
                    var args = ArrayList(Expression).init(allocator);
                    defer args.deinit();

                    var last_arg = false;
                    var finished = false;
                    while (have_tokens(&slice_copy)) {
                        switch (try peek_token(&slice_copy)) {
                            .right_paren => {
                                try consume_token(&slice_copy);
                                var left = try allocator.create(Expression);
                                left.* = expr;

                                expr = Expression{ .call = .{ .left = left, .args = args.toOwnedSlice() } };

                                // Break here, and attempt to chain on another call
                                finished = true;
                                break;
                            },
                            else => {
                                if (last_arg) {
                                    return error.ExpectedClosingParenAfterExpressionWithNoComma;
                                }

                                if (try parse_expression(allocator, &slice_copy)) |arg_expr| {
                                    try args.append(arg_expr);
                                } else {
                                    return error.CouldNotParseExpressionInCall;
                                }

                                switch (try peek_token(&slice_copy)) {
                                    .comma => try consume_token(&slice_copy),
                                    else => last_arg = true,
                                }
                            },
                        }
                    }

                    if (!finished) {
                        return error.RanOutOfTokensWhenParsingCall;
                    }
                },
                .dot => {
                    try consume_token(&slice_copy);

                    var info = try expect_match_token(&slice_copy, .identifier);
                    var name = try copy_slice(allocator, info.identifier.text);
                    errdefer allocator.free(name);

                    var left = try allocator.create(Expression);
                    left.* = expr;

                    expr = Expression{ .attribute = .{ .left = left, .name = name } };
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
                var text = try copy_slice(allocator, string_literal.text);
                return Expression{ .string_literal = text };
            },
            .identifier => |token_ident| {
                try consume_token(tokens);
                var ident = try copy_slice(allocator, token_ident.text);
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

pub fn copy_slice(allocator: *Allocator, slice: []const u8) ![]u8 {
    var new_slice = try allocator.alloc(u8, slice.len);
    std.mem.copy(u8, new_slice, slice);

    return new_slice;
}

// primary        → "true" | "false"
//                | NUMBER | STRING | IDENTIFIER | "(" expression ")";

pub fn have_text(text: *[]const u8) bool {
    return text.*.len >= 1;
}

pub fn have_tokens(tokens: *[]const Token) bool {
    return tokens.*.len >= 1;
}

fn Buffer(comptime T: type) type {
    return struct {
        allocator: *Allocator,
        items: []T,
        capacity: usize,

        fn init(allocator: *Allocator, capacity: usize) !Buffer(T) {
            // Allocate the space for the given capacity, but set the slice
            // to only ever contain the number of items that have been inserted.
            var items = try allocator.alloc(T, capacity);
            items.len = 0;

            return Buffer(T){
                .allocator = allocator,
                .items = items,
                .capacity = capacity,
            };
        }

        fn append(self: *Buffer(T), value: T) !void {
            if (self.items.len < self.capacity) {
                self.items.len += 1;
                self.items[self.items.len - 1] = value;
                return;
            }

            return error.IndexOutOfBounds;
        }

        fn deinit(self: Buffer(T)) void {
            self.allocator.free(self.items);
        }
    };
}

/// Runtime state storage
const Environment = struct {
    allocator: *Allocator,
    ident_map: std.StringHashMap(ShimValue),

    /// Removes this stack layer, returning a lower layer
    fn init(allocator: *Allocator) Environment {
        return .{ .allocator = allocator, .ident_map = std.StringHashMap(ShimValue).init(allocator) };
    }

    fn insert(self: *@This(), ident: []const u8, value: ShimValue) !void {
        if (try self.ident_map.fetchPut(ident, value)) |prev_entry| {
            // If it's already there, the previous key stays. The new key is
            // freed, and the old value is freed...
            self.allocator.free(ident);
            prev_entry.value.deinit(self.allocator);
        }
    }

    /// Return a copy of a value from the environment.
    ///
    /// Caller is responsible for freeing this value.
    fn retrieve(self: @This(), ident: []const u8) !ShimValue {
        if (self.ident_map.get(ident)) |value| {
            // We clone so that the consumer of this value can proceed to free
            // it like all other values.
            // TODO: no tests leak or double-free when value is returned directly
            // so apparently I don't know what's going on.
            return try value.clone(self.allocator);
        }

        // TODO: check in environments up the stack

        return error.UnknownIdentifier;
    }

    fn _deinit_kv(self: @This(), entry: std.StringHashMap(ShimValue).KV) void {
        self.allocator.free(entry.key);
        entry.value.deinit(self.allocator);
    }

    fn _deinit_entry(self: @This(), entry: std.StringHashMap(ShimValue).Entry) void {
        self.allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit(self.allocator);
    }

    fn deinit(self: *@This()) void {
        var iter = self.ident_map.iterator();
        while (iter.next()) |entry| {
            self._deinit_entry(entry);
        }

        self.ident_map.deinit();
    }
};

const BlockExitTag = enum {
    exit_return,
    exit_continue,
    exit_break,
    finished,
};

const BlockExit = union(BlockExitTag) {
    exit_return: ShimValue,
    exit_break: ShimValue,
    exit_continue,
    finished,

    fn deinit(self: @This(), allocator: *Allocator) void {
        switch (self) {
            .exit_return => |val| val.deinit(allocator),
            .exit_break => |val| val.deinit(allocator),
            .exit_continue => {},
            .finished => {},
        }
    }
};

const Interpreter = struct {
    allocator: *Allocator,
    env: Environment,

    pub fn init(allocator: *Allocator) @This() {
        return .{
            .allocator = allocator,
            .env = Environment.init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.env.deinit();
    }

    pub fn interpret_ast(self: *@This(), ast: Ast) !void {
        var result = try self.interpret_block(ast.block);
        result.deinit(self.allocator);
    }

    // TODO: this shouldn't return a ShimValue, it should return a different
    // value depending on whether the value should propagate (return) or
    // not (break, continue, etc.).
    pub fn interpret_block(self: *@This(), block: Block) anyerror!BlockExit {
        for (block.stmts.items) |stmt| {
            // When a statement returns a value, the block is finished
            if (try self.interpret_stmt(stmt)) |val| {
                switch (val) {
                    .exit_return => return val,
                    .exit_break => return val,
                    .exit_continue => return val,
                    .finished => {},
                }
            }
        }

        return BlockExit.finished;
    }

    pub fn interpret_stmt(self: *@This(), stmt: Statement) !?BlockExit {
        switch (stmt) {
            StatementTag.expression_statement => {
                var result = try self.interpret_expr(&stmt.expression_statement);
                result.deinit(self.allocator);
            },
            StatementTag.break_statement => return BlockExit{ .exit_break = ShimValue{ .shim_unit = ShimUnit{} } },
            StatementTag.while_statement => |while_stmt| {
                while (true) {
                    var value = try self.interpret_expr(&while_stmt.predicate);
                    defer value.deinit(self.allocator);

                    if (!value.is_truthy()) {
                        break;
                    }

                    var result = try self.interpret_block(while_stmt.block);
                    defer result.deinit(self.allocator);

                    switch (result) {
                        .exit_return => return result,
                        .exit_break => break,
                        .exit_continue => continue,
                        .finished => {},
                    }
                }
            },
            StatementTag.if_statement => |if_stmt| {
                var value = try self.interpret_expr(&if_stmt.predicate);
                defer value.deinit(self.allocator);

                if (value.is_truthy()) {
                    var result = try self.interpret_block(if_stmt.if_block);
                    switch (result) {
                        .exit_return => return result,
                        .exit_break => return result,
                        .exit_continue => return result,
                        .finished => {},
                    }
                    result.deinit(self.allocator);
                } else if (if_stmt.else_block) |else_block| {
                    var result = try self.interpret_block(else_block);
                    switch (result) {
                        .exit_return => return result,
                        .exit_break => return result,
                        .exit_continue => return result,
                        .finished => {},
                    }
                    result.deinit(self.allocator);
                }
            },
            StatementTag.assignment_statement => |assi| {
                var value = try self.interpret_expr(&assi.value);
                errdefer value.deinit(self.allocator);

                var ident = try copy_slice(self.allocator, assi.name);
                errdefer self.allocator.free(ident);

                try self.env.insert(ident, value);
            },
            StatementTag.declaration_statement => |decl| {
                var value = try self.interpret_expr(&decl.value);
                errdefer value.deinit(self.allocator);

                var key = try copy_slice(self.allocator, decl.name);
                errdefer self.allocator.free(key);

                try self.env.insert(key, value);
            },
        }

        // Since we didn't return a value from the switch, this must not be the
        // sort of statement that breaks out of a block
        return null;
    }

    pub fn interpret_expr(self: @This(), expr: *const Expression) anyerror!ShimValue {
        switch (expr.*) {
            ExpressionTag.int_literal => |v| return ShimValue{ .shim_i128 = v },
            ExpressionTag.string_literal => |str| {
                var new_str = try copy_slice(self.allocator, str);
                return ShimValue{ .shim_str = new_str };
            },
            ExpressionTag.bool_literal => |b| return ShimValue{ .shim_bool = b },
            ExpressionTag.unary => |_| return ShimValue{ .shim_unit = ShimUnit{} },
            ExpressionTag.logical => |_| unreachable,
            ExpressionTag.binary => |bexpr| {
                var left_val = try self.interpret_expr(bexpr.left);
                defer left_val.deinit(self.allocator);
                var right_val = try self.interpret_expr(bexpr.right);
                defer right_val.deinit(self.allocator);

                var left = switch (left_val) {
                    ValueTag.shim_i128 => |v| v,
                    else => return ShimValue{ .shim_unit = ShimUnit{} },
                };
                var right = switch (right_val) {
                    ValueTag.shim_i128 => |v| v,
                    else => return ShimValue{ .shim_unit = ShimUnit{} },
                };
                switch (bexpr.op) {
                    BinaryOperator.BinaryAdd => return ShimValue{ .shim_i128 = left + right },
                    BinaryOperator.BinaryMul => return ShimValue{ .shim_i128 = left * right },
                    BinaryOperator.BinarySub => return ShimValue{ .shim_i128 = left - right },
                    BinaryOperator.BinaryDiv => return ShimValue{ .shim_i128 = @divTrunc(left, right) },
                    BinaryOperator.BinaryDoubleEqual => return ShimValue{ .shim_bool = left == right },
                    BinaryOperator.BinaryBangEqual => return ShimValue{ .shim_bool = left != right },
                    BinaryOperator.BinaryGreaterThanEqual => return ShimValue{ .shim_bool = left >= right },
                    BinaryOperator.BinaryLessThanEqual => return ShimValue{ .shim_bool = left <= right },
                    BinaryOperator.BinaryLessThan => return ShimValue{ .shim_bool = left < right },
                    BinaryOperator.BinaryGreaterThan => return ShimValue{ .shim_bool = left > right },
                }
            },
            ExpressionTag.identifier => |ident| {
                if (std.mem.eql(u8, ident, "print")) {
                    return ShimValue.create_native_fn(&native_fn_print);
                }
                // For now, just error when an ident isn't there
                return try self.env.retrieve(ident);
            },
            ExpressionTag.attribute => |aexpr| {
                var left = try self.interpret_expr(aexpr.left);
                defer left.deinit(self.allocator);

                return try left.get(self.allocator, aexpr.name);
            },
            ExpressionTag.call => |cexpr| {
                var left = try self.interpret_expr(cexpr.left);
                defer left.deinit(self.allocator);

                var args = try Buffer(ShimValue).init(self.allocator, cexpr.args.len);
                defer {
                    for (args.items) |arg| {
                        arg.deinit(self.allocator);
                    }
                    args.deinit();
                }

                for (cexpr.args) |arg_expr| {
                    var arg_val = try self.interpret_expr(&arg_expr);
                    try args.append(arg_val);
                }

                return left.call(self.allocator, args.items);
            },
        }
    }
};

fn native_fn_print(allocator: *Allocator, values: []const ShimValue) anyerror!ShimValue {
    _ = allocator;
    // We take a pointer to workaround a bug I still need to file: https://www.reddit.com/r/zig/comments/py099g
    for (values) |*val, i| {
        switch (val.*) {
            .shim_i128 => |num| std.debug.print("{}", .{num}),
            .shim_str => |str| std.debug.print("{s}", .{str}),
            .shim_bool => |b| std.debug.print("{s}", .{b}),
            else => std.debug.print("<{}>", .{val.*}),
        }
        if (i != values.len - 1) {
            std.debug.print(" ", .{});
        }
    }
    std.debug.print("\n", .{});

    return ShimValue{ .shim_unit = ShimUnit{} };
}

fn native_str_lower(allocator: *Allocator, values: []const ShimValue) anyerror!ShimValue {
    _ = allocator;
    if (values.len != 1) {
        return ShimValue{ .shim_error = ShimUnit{} };
    }

    switch (values[0]) {
        .shim_str => |str| {
            return ShimValue{ .shim_str = try std.ascii.allocLowerString(allocator, str) };
        },
        else => return error.HowWasLowerCalledOnNonStr,
    }
}
