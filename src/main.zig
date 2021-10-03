const std = @import("std");
const libshim = @import("libshim");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked) {
            std.log.err("Leaked: {}", .{leaked});
            std.os.exit(42);
        }
    }

    var args = std.process.args();

    _ = args.nextPosix();
    if (args.nextPosix()) |script_path| {
        var script_file = std.fs.cwd().openFile(script_path, std.fs.File.OpenFlags{ .read = true }) catch {
            std.debug.print("Couldn't find file {s}\n", .{script_path});
            return;
        };
        defer script_file.close();

        var five_megabytes: u64 = 5 * 1024 * 1024;
        var script_text = script_file.readToEndAlloc(&gpa.allocator, five_megabytes) catch {
            std.debug.print("Error reading file {s}\n", .{script_path});
            return;
        };
        defer gpa.allocator.free(script_text);

        libshim.run_text(&gpa.allocator, script_text) catch |err| {
            std.debug.print("{}\n", .{err});
        };
        return;
    }

    std.debug.print("No script provided\n", .{});
}
