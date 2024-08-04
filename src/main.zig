const std = @import("std");

const parser = @import("parser.zig");

fn get_script_name(allocator: std.mem.Allocator) !?([:0]const u8) {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var is_first_iteration = true;
    while (true) {
        if (args.next()) |arg| {
            if (!is_first_iteration) {
                // TODO: Seems like this should be copied out? Potentially freed
                // when `args` is `deinit`'d.
                return try allocator.dupeZ(u8, arg);
            }
            is_first_iteration = false;
        } else {
            break;
        }
    }

    return null;
}

const OperandFormat = enum(u8) {
    hex,
    u,
    s,
};

const OperandInfo = struct {
    format: [4]?OperandFormat,
};

pub fn pushByte(buffer: *[]u8, byte: u8) !void {
    if (buffer.len == 0) {
        return error.BytesExhausted;
    }
    buffer.*[0] = byte;
    buffer.* = buffer.*[1..];
}

const OpCode = enum(u8) {
    noop,
    push_literal,
    push_null,
    push_short_str,
    add,
    print,
    jmpnz,
    sub,
    end,
    jmp,
    pop,
    set_env,
    pushInt,
    pushFloat,
    pushTrue,
    pushFalse,
    createList,
    get_env,
    call,
    index,

    pub fn operandCount(self: OpCode) usize {
        const ops = self.operandInfo();
        if (ops.format[3] != null) {
            return 4;
        }
        if (ops.format[2] != null) {
            return 3;
        }
        if (ops.format[1] != null) {
            return 2;
        }
        if (ops.format[0] != null) {
            return 1;
        } else {
            return 0;
        }
    }

    pub fn name(self: OpCode) [:0]const u8 {
        return @tagName(self);
    }

    pub fn push(comptime self: OpCode, bytes: *[]u8, operands: anytype) !void {
        if (bytes.len == 0) {
            return error.BytesExhausted;
        }

        try pushByte(bytes, @intFromEnum(self));

        switch (self) {
            OpCode.sub => {},
            OpCode.print => {},
            OpCode.end => {},
            OpCode.push_literal => {
                try pushByte(bytes, operands[0]);
            },
            OpCode.push_short_str => {
                if (operands[0].len > 255) {
                    return error.StringTooLong;
                }

                try pushByte(bytes, @intCast(operands[0].len));
                for (operands[0]) |chr| {
                    try pushByte(bytes, chr);
                }
            },
            OpCode.jmpnz => {
                try pushByte(bytes, operands[0]);
            },
            OpCode.set_env => {},
            OpCode.push_null => {},
            OpCode.add => {},
            OpCode.pop => {},

            OpCode.pushInt => {
                const b: u64 = @bitCast(operands[0]);
                try pushByte(bytes, @intCast((b & 0xff000000) >> 24));
                try pushByte(bytes, @intCast((b & 0x00ff0000) >> 16));
                try pushByte(bytes, @intCast((b & 0x0000ff00) >> 8));
                try pushByte(bytes, @intCast((b & 0x000000ff) >> 0));
            },
            OpCode.pushFloat => {
                const b: u64 = @bitCast(operands[0]);
                try pushByte(bytes, @intCast((b & 0xff000000) >> 24));
                try pushByte(bytes, @intCast((b & 0x00ff0000) >> 16));
                try pushByte(bytes, @intCast((b & 0x0000ff00) >> 8));
                try pushByte(bytes, @intCast((b & 0x000000ff) >> 0));
            },
            OpCode.pushTrue => {},
            OpCode.pushFalse => {},
            OpCode.createList => {
                if (operands[0] > 255) {
                    return error.ListTooLong;
                }
                try pushByte(bytes, @intCast(operands[0]));
            },
            OpCode.get_env => {},
            OpCode.call => {
                if (operands[0] > 255) {
                    return error.ListTooLong;
                }
                try pushByte(bytes, @intCast(operands[0]));
            },
            OpCode.index => {},

            else => {
                @compileLog("OpCode.push not implemented for ", self);
                @compileError("Unimplemented opcode");
            },
        }
    }
};

pub fn bc_jmpnz_abs(bytes: []u8, pc: *usize, address: usize) void {
    const count: isize = @intCast(-@as(isize, @intCast(pc.*)) + @as(isize, @intCast(address)) - 2);

    const count_i8: i8 = @intCast(count);
    const count_u8: u8 = @bitCast(count_i8);

    OpCode.jmpnz.push(bytes, pc, .{count_u8});
}

pub fn compileExpression(expr: parser.Expression, bytes: *[]u8) !void {
    switch (expr) {
        parser.ExpressionTag.plus => |op| {
            try compileExpression(op.a.*, bytes);
            try compileExpression(op.b.*, bytes);
            try OpCode.add.push(bytes, .{});
        },
        parser.ExpressionTag.primary => |prim| {
            switch (prim) {
                parser.PrimaryTag.intLiteral => |i| {
                    try OpCode.pushInt.push(bytes, .{i});
                },
                parser.PrimaryTag.floatLiteral => |f| {
                    try OpCode.pushFloat.push(bytes, .{f});
                },
                parser.PrimaryTag.boolLiteral => |b| {
                    if (b) {
                        try OpCode.pushTrue.push(bytes, .{});
                    } else {
                        try OpCode.pushFalse.push(bytes, .{});
                    }
                },
                parser.PrimaryTag.nullLiteral => {
                    try OpCode.push_null.push(bytes, {});
                },
                parser.PrimaryTag.stringLiteral => |str| {
                    if (str.len > 255) {
                        return error.StringTooLong;
                    }
                    try OpCode.push_short_str.push(bytes, .{str});
                },
                parser.PrimaryTag.listLiteral => |lst| {
                    for (lst) |lstExpr| {
                        try compileExpression(lstExpr, bytes);
                    }
                    try OpCode.createList.push(bytes, .{lst.len});
                },
                parser.PrimaryTag.identifier => |id| {
                    try OpCode.push_short_str.push(bytes, .{id});
                    try OpCode.get_env.push(bytes, .{});
                },
            }
        },
        parser.ExpressionTag.call => |call_info| {
            try compileExpression(call_info.callee.*, bytes);
            for (call_info.args) |arg_expr| {
                try compileExpression(arg_expr, bytes);
            }
            try OpCode.call.push(bytes, .{call_info.args.len});
        },
        parser.ExpressionTag.index => |index_info| {
            try compileExpression(index_info.callee.*, bytes);
            try compileExpression(index_info.index.*, bytes);
            try OpCode.index.push(bytes, .{});
        },
        else => {
            std.debug.print("line: {}  {} not implemented\n", .{ @src().line, @as(parser.ExpressionTag, expr) });
            return error.NotImplemented;
        },
    }
}

pub fn compileDeclaration(decl: parser.Declaration, bytes: *[]u8) !void {
    switch (decl) {
        parser.DeclarationTag.variable_declaration => |var_decl| {
            if (var_decl.expr) |expr| {
                try compileExpression(expr, bytes);
            } else {
                try OpCode.push_null.push(bytes, .{});
            }
            try OpCode.push_short_str.push(bytes, .{var_decl.name});
            try OpCode.set_env.push(bytes, .{});
        },
        parser.DeclarationTag.statement_declaration => |stmt| {
            switch (stmt) {
                parser.StatementTag.expression_statement => |expr| {
                    try compileExpression(expr.*, bytes);
                    // Need to pop the value pushed by compileExpression...
                    // Seems like a waste to push a value only to pop it though...
                    try OpCode.pop.push(bytes, .{});
                },
                else => {
                    std.debug.print("line: {}  {} not implemented\n", .{ @src().line, @as(parser.StatementTag, stmt) });
                    return error.NotImplemented;
                },
            }
        },
        else => {
            std.debug.print("line: {}  {} not implemented\n", .{ @src().line, @as(parser.DeclarationTag, decl) });
            return error.NotImplemented;
        },
    }
}

fn u64FromBytes(bytes: [4]u8) u64 {
    const a: u64 = @intCast(bytes[0]);
    const b: u64 = @intCast(bytes[1]);
    const c: u64 = @intCast(bytes[2]);
    const d: u64 = @intCast(bytes[3]);
    return (a << 24) + (b << 16) + (c << 8) + d;
}

fn i64FromBytes(bytes: [4]u8) i64 {
    const b = u64FromBytes(bytes);
    return @bitCast(b);
}

fn f64FromBytes(bytes: [4]u8) f64 {
    const b = u64FromBytes(bytes);
    return @bitCast(b);
}

pub fn printBytecode(bytes: []const u8) void {
    var idx: usize = 0;
    while (idx < bytes.len) {
        const opcode: OpCode = @enumFromInt(bytes[idx]);

        std.debug.print("{x:0>5}: ", .{idx});
        std.debug.print("{any}", .{opcode});
        idx += 1;

        switch (opcode) {
            OpCode.push_literal => {
                std.debug.print(" '{c}'", .{bytes[idx]});
                idx += 1;
            },
            OpCode.push_short_str => {
                const len = bytes[idx];
                idx += 1;

                const slice = bytes[idx .. idx + len];
                idx += len;

                std.debug.print(" {any} {s}", .{ len, slice });
            },
            OpCode.pushInt => {
                const slice: [4]u8 = .{ bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3] };
                const b = u64FromBytes(slice);
                idx += 4;

                std.debug.print(" {any}", .{@as(i64, @bitCast(b))});
            },
            OpCode.pushFloat => {
                const slice: [4]u8 = .{ bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3] };
                const f = f64FromBytes(slice);
                idx += 4;

                std.debug.print(" {any}", .{f});
            },
            OpCode.createList => {
                const len = bytes[idx];
                idx += 1;
                std.debug.print(" {any}", .{len});
            },
            OpCode.call => {
                const len = bytes[idx];
                idx += 1;
                std.debug.print(" {any}", .{len});
            },
            OpCode.index => {},
            OpCode.noop => {},
            OpCode.add => {},
            OpCode.print => {},
            OpCode.jmpnz => {},
            OpCode.sub => {},
            OpCode.end => {},
            OpCode.jmp => {},
            OpCode.pop => {},
            OpCode.push_null => {},
            OpCode.set_env => {},
            OpCode.pushTrue => {},
            OpCode.pushFalse => {},
            OpCode.get_env => {},
        }

        std.debug.print("\n", .{});
    }
}

pub fn compileAst(allocator: std.mem.Allocator, ast: parser.Ast) ![]const u8 {
    const max_bytes = 1 << 16;
    var buffer = try allocator.alloc(u8, max_bytes);
    for (0..max_bytes) |idx| {
        buffer[idx] = @intFromEnum(OpCode.noop);
    }

    var bytes = buffer;
    for (ast.declarations) |decl| {
        try compileDeclaration(decl, &bytes);
    }

    // End immediately
    try OpCode.end.push(&bytes, .{});

    // Only return the bytes that were written to
    return buffer[0..(max_bytes - bytes.len)];
}

const ShimValueTag = enum(u8) {
    str,
    int,
    float,
    nullValue,
    boolean,
    printFunction,
    list,
};

const ShimValue = union(ShimValueTag) {
    str: []const u8,
    int: i64,
    float: f64,
    nullValue: void,
    boolean: bool,
    printFunction: void,
    list: std.ArrayList(ShimValue),
};

fn pushInt(stack: *std.ArrayList(ShimValue), val: i64) !void {
    try stack.append(.{ .int = val });
}

fn popInt(stack: *std.ArrayList(ShimValue)) !i64 {
    const value = stack.pop();
    switch (value) {
        ShimValue.int => |i| {
            return i;
        },
        else => {
            return error.ExpectedInt;
        },
    }
}

fn read_u8(bytecode: []const u8, idx: *usize) u8 {
    const val = bytecode[idx.*];
    idx.* = idx.* + 1;
    return val;
}

fn printStack(stack: std.ArrayList(ShimValue)) void {
    if (stack.items.len == 0) {
        std.debug.print("stack: empty\n", .{});
    } else {
        std.debug.print("stack:\n", .{});
        std.debug.print("  BOTTOM OF STACK\n", .{});
        for (stack.items) |val| {
            std.debug.print("    {any}\n", .{val});
        }
        std.debug.print("  TOP OF STACK\n", .{});
    }
}

fn printEnv(env: std.StringHashMap(ShimValue)) void {
    std.debug.print("env:\n", .{});

    var it = env.iterator();
    while (it.next()) |val| {
        std.debug.print("    {s}: {any}\n", .{ val.key_ptr.*, val.value_ptr.* });
    }
}

const ShimState = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !ShimState {
        return ShimState{ .allocator = allocator };
    }

    pub fn deinit(_: ShimState) void {}

    pub fn loadBytecode(self: ShimState, bytecode: []const u8) !void {
        var env = std.StringHashMap(ShimValue).init(self.allocator);

        try env.put("print", .{ .printFunction = {} });

        var pc: usize = 0;
        var stack = std.ArrayList(ShimValue).init(self.allocator);
        while (pc < bytecode.len) {
            const op: OpCode = @enumFromInt(read_u8(bytecode, &pc));
            switch (op) {
                OpCode.push_literal => {
                    try pushInt(&stack, @intCast(bytecode[pc]));
                },

                OpCode.push_null => {
                    try stack.append(.{ .nullValue = {} });
                },
                OpCode.push_short_str => {
                    const len = read_u8(bytecode, &pc);

                    const str = bytecode[pc .. pc + len];
                    try stack.append(.{ .str = str });

                    pc += len;
                },
                OpCode.set_env => {
                    const name = stack.pop();
                    const value = stack.pop();

                    switch (name) {
                        ShimValueTag.str => |str| {
                            try env.put(str, value);
                        },
                        else => {
                            return error.SetEnvOnNonStr;
                        },
                    }
                },

                OpCode.pop => {
                    _ = stack.pop();
                },
                OpCode.add => {
                    const b = try popInt(&stack);
                    const a = try popInt(&stack);
                    const c = a + b;
                    try pushInt(&stack, c);
                },
                OpCode.sub => {
                    const b = try popInt(&stack);
                    const a = try popInt(&stack);
                    const c = a - b;
                    try pushInt(&stack, c);
                },
                OpCode.jmpnz => {
                    const inst_count: i8 = @bitCast(bytecode[pc]);

                    const val = try popInt(&stack);
                    if (val != 0) {
                        pc = @intCast(@as(isize, @intCast(pc)) + inst_count);
                    }
                },
                OpCode.jmp => {
                    const inst_count: i8 = @bitCast(bytecode[pc]);

                    pc = @intCast(@as(isize, @intCast(pc)) + inst_count);
                },
                OpCode.print => {
                    std.debug.print("{any}", .{stack.pop()});
                },
                OpCode.noop => {},

                OpCode.pushInt => {
                    const i = i64FromBytes(.{
                        bytecode[pc + 0],
                        bytecode[pc + 1],
                        bytecode[pc + 2],
                        bytecode[pc + 3],
                    });
                    pc += 4;
                    try stack.append(.{ .int = i });
                },

                OpCode.pushFloat => {
                    const f = f64FromBytes(.{
                        bytecode[pc + 0],
                        bytecode[pc + 1],
                        bytecode[pc + 2],
                        bytecode[pc + 3],
                    });
                    pc += 4;
                    try stack.append(.{ .float = f });
                },

                OpCode.createList => {
                    const len = read_u8(bytecode, &pc);
                    var lst = std.ArrayList(ShimValue).init(self.allocator);
                    for (0..len) |idx| {
                        try lst.append(stack.items[stack.items.len - len + idx]);
                    }
                    stack.shrinkAndFree(stack.items.len - len);
                    try stack.append(.{ .list = lst });
                },

                OpCode.call => {
                    const len = read_u8(bytecode, &pc);
                    var lst = std.ArrayList(ShimValue).init(self.allocator);
                    for (0..len) |idx| {
                        try lst.append(stack.items[stack.items.len - len + idx]);
                    }
                    stack.shrinkAndFree(stack.items.len - len);

                    const callee = stack.pop();
                    switch (callee) {
                        ShimValue.printFunction => {
                            std.debug.print("{}\n", .{lst});
                            try stack.append(.{ .nullValue = {} });
                        },
                        else => {
                            std.debug.print("Tried to call {}\n", .{callee});
                            return error.CalledNonPrintFunction;
                        },
                    }
                },
                OpCode.index => {
                    const index = stack.pop();
                    const callee = stack.pop();
                    switch (callee) {
                        ShimValue.list => |lst| {
                            switch (index) {
                                ShimValue.int => |i| {
                                    if (i >= lst.items.len) {
                                        return error.IndexOutOfBound;
                                    }
                                    if (i < 0) {
                                        return error.IndexedWithNegativeInt;
                                    }
                                    const idx: usize = @intCast(i);
                                    try stack.append(lst.items[idx]);
                                },
                                else => {
                                    std.debug.print("Tried to index with {}\n", .{index});
                                    return error.IndexValueNonInt;
                                },
                            }
                        },
                        else => {
                            std.debug.print("Tried to index {}\n", .{callee});
                            return error.CalledNonPrintFunction;
                        },
                    }
                },

                OpCode.pushTrue => {
                    try stack.append(.{ .boolean = true });
                },
                OpCode.pushFalse => {
                    try stack.append(.{ .boolean = false });
                },
                OpCode.get_env => {
                    const name = stack.pop();

                    switch (name) {
                        ShimValueTag.str => |str| {
                            if (env.get(str)) |val| {
                                try stack.append(val);
                            } else {
                                std.debug.print("Undefined name {s}\n", .{str});
                                return error.NameUndefined;
                            }
                        },
                        else => {
                            return error.GetEnvOnNonStr;
                        },
                    }
                },

                OpCode.end => {
                    printStack(stack);
                    printEnv(env);
                    break;
                },
            }
        }

        if (stack.items.len != 0) {
            std.debug.print("Items left on stack! Probably a bug?\n", .{});
        }
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var scriptName: [:0]const u8 = undefined;
    if (try get_script_name(allocator)) |name| {
        scriptName = name;
    } else {
        std.debug.print("Shimlang: No script provided\n", .{});
        return;
    }

    // std.debug.print("Script name {s}\n", .{scriptName});
    defer allocator.free(scriptName);

    const cwd = std.fs.cwd();
    const file = try cwd.openFile(scriptName, .{ .mode = .read_only });
    defer file.close();

    const text = try file.readToEndAlloc(allocator, 1_000_000);
    defer allocator.free(text);

    // std.debug.print("File text:\n{s}\n", .{text});

    // const tokens = try tokenize(allocator, text);
    // std.debug.print("Tokens {any}\n", .{tokens});

    const ast = try parser.parseProgramText(allocator, text);
    defer ast.deinit();

    ast.print(allocator);

    const bytecode = try compileAst(allocator, ast);
    defer allocator.free(bytecode);

    printBytecode(bytecode);

    var state = try ShimState.init(allocator);
    defer state.deinit();

    try state.loadBytecode(bytecode);
}
