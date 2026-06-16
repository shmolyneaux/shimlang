# Shimlang Language Reference

Shimlang is a dynamically-typed scripting language designed for the SHIM game
engine. It features a clean syntax, first-class functions with closures, structs
with methods, and a growing standard library of built-in types and functions.

> **Note:** Shimlang is under active development. This document describes the
> current state of the language as it exists today.

## Table of Contents

- [Hello, World!](#hello-world)
- [Variables](#variables)
- [Data Types](#data-types)
  - [Integers](#integers)
  - [Floats](#floats)
  - [Booleans](#booleans)
  - [Strings](#strings)
  - [None](#none)
  - [Lists](#lists)
  - [Tuples](#tuples)
  - [Dictionaries](#dictionaries)
  - [Numeric Methods](#numeric-methods)
- [Operators](#operators)
  - [Arithmetic](#arithmetic)
  - [Comparison](#comparison)
  - [Membership](#membership)
  - [Logical](#logical)
  - [Negation](#negation)
  - [Range](#range)
  - [Assignment Operators](#assignment-operators)
- [Control Flow](#control-flow)
  - [If / Else / Else If](#if--else--else-if)
  - [While Loops](#while-loops)
  - [For Loops](#for-loops)
  - [Break and Continue](#break-and-continue)
- [Functions](#functions)
  - [Defining Functions](#defining-functions)
  - [Implicit Return](#implicit-return)
  - [Default Arguments](#default-arguments)
  - [Keyword Arguments](#keyword-arguments)
  - [Anonymous Functions](#anonymous-functions)
  - [Closures](#closures)
- [Structs](#structs)
  - [Defining Structs](#defining-structs)
  - [Methods](#methods)
  - [Static Methods](#static-methods)
  - [Default Field Values](#default-field-values)
  - [Operator Overloading](#operator-overloading)
  - [Struct Introspection](#struct-introspection)
- [String Interpolation](#string-interpolation)
- [Block Expressions](#block-expressions)
- [Statement Terminators](#statement-terminators)
- [Comments](#comments)
- [Custom Iterators](#custom-iterators)
- [Built-in Functions](#built-in-functions)
- [Error Handling Philosophy](#error-handling-philosophy)
- [Suggested Improvements](#suggested-improvements)

## Hello, World!

The simplest Shimlang program prints a message to the console using the built-in
`print` function:

```rust
print("Hello, World!");
```

Output:

```
Hello, World!
```

`print` accepts multiple arguments separated by commas, which are printed
space-separated:

```rust
print("The answer is", 42);
```

Output:

```
The answer is 42
```

## Variables

Variables are declared with `let` and can hold any type. Reassignment uses `=`
without the `let` keyword:

```rust
let greeting = "Hello";
let year = 2024;
print(greeting, year);

let x = 10;
x = 20;
print(x);
```

Output:

```
Hello 2024
20
```

## Data Types

### Integers

Integers are 32-bit signed values. They support standard arithmetic operations:

```rust
let a = 42;
let b = -7;
print(a);
print(b);
print(a + b);
```

Output:

```
42
-7
35
```

### Floats

Floating-point numbers are 32-bit (single precision). A number with a decimal
point is treated as a float:

```rust
let pi = 3.14;
let neg = -2.5;
print(pi);
print(neg);
```

Output:

```
3.14
-2.5
```

The `/` operator always produces a float, even for integer operands. Use
`.trunc()` or `int(...)` if you need a truncated integer result:

```rust
print(10 / 3);
print(10.0 / 3.0);
print(int(10 / 3));
```

Output:

```
3.3333333
3.3333333
3
```

### Booleans

The two boolean literals are `true` and `false`:

```rust
print(true);
print(false);
```

Output:

```
true
false
```

### Strings

Strings are created with double quotes. They support indexing, length, escape
sequences, and concatenation with `+`:

```rust
let name = "Shimlang";
print(name);
print(name.len());
print(name[0]);
print("Hello" + " " + "World");
```

Output:

```
Shimlang
8
S
Hello World
```

Supported escape sequences include `\n` (newline), `\t` (tab), `\"` (double
quote), and `\\` (backslash):

```rust
print("line one\nline two");
print("a \"quoted\" word");
```

Output:

```
line one
line two
a "quoted" word
```

Strings are byte-oriented and currently intended for ASCII text. `.len()`
returns the number of bytes, indexing returns a one-byte string, and iteration
yields each ASCII character byte as a one-character string:

```rust
for ch in "abc" {
    print(ch);
}
```

Output:

```
a
b
c
```

Strings can be compared for equality:

```rust
let a = "test";
let b = "test";
let c = "other";
print(a == b);
print(a == c);
```

Output:

```
true
false
```

Strings expose these methods:

| Method | Description |
|--------|-------------|
| `.len()` | Returns the byte length |
| `.split()` | Splits on ASCII whitespace and omits empty fields |
| `.join(iterable)` | Joins the string representations of `iterable` with this string as the separator |
| `.upper()` | Returns an ASCII-uppercase copy |
| `.lower()` | Returns an ASCII-lowercase copy |
| `.strip()` | Trims ASCII whitespace from both ends |
| `.lstrip()` | Trims ASCII whitespace from the left |
| `.rstrip()` | Trims ASCII whitespace from the right |
| `.remove_prefix(prefix)` | Removes `prefix` if present |
| `.remove_suffix(suffix)` | Removes `suffix` if present |
| `.split_lines()` | Splits on `\n`, `\r\n`, or `\r` line endings |
| `.contains(needle)` | Returns whether `needle` appears in the string |
| `.starts_with(prefix)` | Returns whether the string starts with `prefix` |
| `.ends_with(suffix)` | Returns whether the string ends with `suffix` |
| `.find(needle)` | Returns the byte index of `needle`, or `None` |
| `.replace(old, new)` | Replaces all occurrences of `old` with `new` |

```rust
let text = "  hello world  ";
print(text.strip());
print(text.contains("world"));
print(",".join(["a", "b", "c"]));
print("one\ntwo".split_lines());
```

Output:

```
hello world
true
a,b,c
[one, two]
```

`replace(old, new)` requires `old` to be non-empty.

### None

`None` represents the absence of a value. Functions that do not explicitly
return a value return `None`:

```rust
let x = None;
print(x);
```

Output:

```
None
```

### Lists

Lists are ordered, mutable collections that can hold values of any type:

```rust
let numbers = [1, 2, 3, 4, 5];
print(numbers);
print(numbers[0]);
print(numbers[-1]);
print(numbers.len());
```

Output:

```
[1, 2, 3, 4, 5]
1
5
5
```

Negative indices count from the end of the list. List elements can be reassigned
by index:

```rust
let lst = ["a", "b", "c"];
lst[0] = "x";
lst[-1] = "z";
print(lst);
```

Output:

```
[x, b, z]
```

Lists have a rich set of methods:

| Method | Description |
|--------|-------------|
| `.len()` | Returns the number of elements |
| `.append(val)` | Adds an element to the end |
| `.pop()` | Removes and returns the last element |
| `.pop(idx)` | Removes and returns the element at `idx` |
| `.insert(idx, val)` | Inserts `val` before position `idx` |
| `.extend(iterable)` | Appends all elements from an iterable |
| `.clear()` | Removes all elements |
| `.index(val)` | Returns the index of `val`, or `None` |
| `.index(val, default)` | Returns the index of `val`, or `default` |
| `.sort()` | Sorts the list in place |
| `.sort(key_fn)` | Sorts in place using a key function |
| `.sorted()` | Returns a new sorted list |
| `.reverse()` | Reverses the list in place |
| `.reversed()` | Returns a new reversed list |
| `.map(fn)` | Returns a new list with `fn` applied to each element |
| `.filter()` | Returns a new list of truthy elements |
| `.filter(fn)` | Returns a new list of elements where `fn` returns truthy |
| `.enumerate()` | Returns an iterable yielding `(index, element)` tuples |
| `.average()` | Returns the arithmetic average of the elements, or `0` for an empty list |

Examples:

```rust
let lst = [3, 1, 4, 1, 5];
lst.append(9);
print(lst);

lst.sort();
print(lst);

let doubled = lst.map(fn(x) { x * 2 });
print(doubled);

let big = lst.filter(fn(x) { x > 3 });
print(big);

let truthy = [0, 1, "", "ok"].filter();
print(truthy);
```

Output:

```
[3, 1, 4, 1, 5, 9]
[1, 1, 3, 4, 5, 9]
[2, 2, 6, 8, 10, 18]
[4, 5, 9]
[1, ok]
```

The `sorted()` method returns a copy without modifying the original:

```rust
let lst = [5, 2, 8, 1];
let sorted_copy = lst.sorted();
print(lst);
print(sorted_copy);
```

Output:

```
[5, 2, 8, 1]
[1, 2, 5, 8]
```

The `sort` method accepts an optional key function. Sorting is stable: elements
with equal keys keep their original relative order. If two sort keys cannot be
compared, those elements keep their original relative order.

```rust
let lst = [5, 2, 8, 1, 9, 3];
lst.sort(fn(x) { 0 - x });
print(lst);
```

Output:

```
[9, 8, 5, 3, 2, 1]
```

The `enumerate()` method pairs each element with its index. It returns an
iterable value, so it is typically consumed by a `for` loop using tuple
unpacking (see [Tuples](#tuples)):

```rust
let names = ["a", "b", "c"];
for i, name in names.enumerate() {
    print(i, name);
}
```

Output:

```
0 a
1 b
2 c
```

To advance it manually, call `.iter()` first and then call `.next()` on the
returned iterator:

```rust
let enumerated = ["a", "b"].enumerate();
let iter = enumerated.iter();
print(iter.next());
print(iter.next());
print(iter.next());
```

Output:

```
(0, a)
(1, b)
None
```

### Tuples

Tuples are fixed-size, ordered, immutable sequences written with parentheses.
Unlike lists, tuples are hashable, so they can be used as dictionary keys:

```rust
let pair = (1, 2);
let triple = (1, "two", 3.0);
print(pair);
print(triple);
print(pair[0], pair[1]);
```

Output:

```
(1, 2)
(1, two, 3)
1 2
```

A single-element tuple requires a trailing comma to distinguish it from a
parenthesized expression. The empty tuple is written `()`:

```rust
let one = (1,);
let empty = ();
print(one);
print(empty);
```

Output:

```
(1,)
()
```

Tuples can be unpacked in `for` loops by listing multiple variables before `in`.
This is the primary way to destructure tuples — there is no `let (x, y) = ...`
form:

```rust
let points = [(1, 2), (3, 4), (5, 6)];
for x, y in points {
    print("\(x) + \(y) = \(x + y)");
}
```

Output:

```
1 + 2 = 3
3 + 4 = 7
5 + 6 = 11
```

Tuples are hashable and may be used as dictionary keys:

```rust
let d = dict();
d[(0, 0)] = "origin";
d[(1, 2)] = "point";
print(d[(0, 0)]);
print((1, 2) in d);
```

Output:

```
origin
true
```

Tuples compare lexicographically. If one tuple is a prefix of another, the
shorter one sorts first:

```rust
let lst = [(2, 1), (1, 2), (1, 1)];
print(lst.sorted());
```

Output:

```
[(1, 1), (1, 2), (2, 1)]
```

### Dictionaries

Dictionaries are hash maps created with the `dict()` built-in. Keys can be
hashable values: integers, floats, booleans, strings, `None`, and tuples whose
items are also hashable:

```rust
let d = dict();
d["name"] = "Alice";
d["age"] = 30;
print(d["name"]);
print(d["age"]);
print(d.len());
```

Output:

```
Alice
30
2
```

You can also use the `.set()` and `.get()` methods:

```rust
let d = dict();
d.set("key", "value");
print(d.get("key"));
```

Output:

```
value
```

`dict()` also accepts keyword arguments. Keyword names become string keys:

```rust
let d = dict(name="Alice", age=30);
print(d["name"], d["age"]);
```

Output:

```
Alice 30
```

Dictionary methods:

| Method | Description |
|--------|-------------|
| `.len()` | Returns the number of entries |
| `.get(key)` | Returns the value for `key`, or `None` if missing |
| `.get(key, default)` | Returns the value for `key`, or `default` if missing |
| `.get(key, default=val)` | Returns the value for `key`, or `val` if missing |
| `.set(key, val)` | Sets `key` to `val` |
| `.has(key)` | Returns `true` if `key` exists |
| `.pop(key)` | Removes `key` and returns its value |
| `.pop(key, default)` | Removes `key` or returns `default` |
| `.keys()` | Returns an iterable over keys |
| `.values()` | Returns an iterable over values |
| `.items()` | Returns an iterable of `(key, value)` pairs |
| `.shrink_to_fit()` | Rebuilds internal storage to fit the current entries |

Iterating over a dictionary yields its keys. Use `.items()` for key-value pairs:

```rust
let d = dict();
d["a"] = 1;
d["b"] = 2;
d["c"] = 3;

for key in d {
    print(key);
}

for key, value in d.items() {
    print(key, "=>", value);
}
```

Output:

```
a
b
c
a => 1
b => 2
c => 3
```

### Numeric Methods

Both integers and floats expose the same set of math methods. Methods that
produce fractional results return floats even when called on an integer:

```rust
print((9).sqrt());
print((-3).abs());
print((2.5).round());
```

Output:

```
3.0
3
3
```

> **Note:** A method call on a negative integer literal needs parentheses
> (`(-3).abs()`), otherwise the `.` binds to the literal `3` first.

Available methods on both `int` and `float`:

| Category | Methods |
|----------|---------|
| Conversion | `bool`, `int`, `float` |
| Sign / range | `abs`, `signum`, `min`, `max`, `clamp`, `in_range` |
| Rounding | `round`, `ceil`, `floor`, `trunc`, `frac` |
| Power / root | `sqrt`, `pow(exp)`, `recip` |
| Logs / exp | `ln`, `log(base)`, `log2`, `log10` |
| Trig | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(other)` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Angle | `to_degrees`, `to_radians` |

`pow`, `log`, and `atan2` take a second operand, which can be passed
positionally or as a keyword argument (`exp`, `base`, `other`):

```rust
print((2).pow(10));
print((1000.0).log(base=10));
print((1.0).atan2(1.0));
```

Output:

```
1024
3.0
0.7853982
```

`min`, `max`, `clamp`, and `in_range` take range or bound operands:

```rust
print((10).min(3));
print((10).max(3));
print((10).clamp(0, 5));
print((3.5).in_range(0.0, 5.0));
```

Output:

```
3
10
5
true
```

## Operators

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition (also string concatenation) |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division (always produces a float) |
| `%` | Modulus (Euclidean: result has the sign of the divisor) |

```rust
print(10 + 3);
print(10 - 3);
print(10 * 3);
print(10 / 3);
print(10 % 3);
```

Output:

```
13
7
30
3.3333333
1
```

Modulus uses Euclidean remainder, so the result always has the same sign as the
divisor. With a positive divisor the result is non-negative:

```rust
print(-1 % 5);
print(-7 % 3);
```

Output:

```
4
2
```

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal to |
| `!=` | Not equal to |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

```rust
print(1 < 2);
print(2 > 1);
print(1 <= 1);
print(1 >= 2);
print(1 == 1);
print(1 != 2);
print(1 != 1);
```

Output:

```
true
true
true
false
true
true
false
```

Ordering comparisons are defined for numbers (integers and floats compare
with each other), strings, booleans, `None`, lists, tuples, and structs that
provide comparison overload methods. Lists and tuples compare lexicographically:
the first unequal element determines the result, and if one sequence is a
prefix of the other, the shorter sequence sorts first.

Equality is defined for booleans, numbers, strings, `None`, lists, tuples,
function identity, bound method identity, struct definitions, and structs.
Struct equality uses an `eq(self, other)` method when present; without it,
distinct struct instances compare unequal.

### Membership

The `in` operator tests containment in dictionaries, lists, and strings:

```rust
let d = dict();
d["x"] = 1;
print("x" in d);
print("y" in d);

let lst = [1, 2, 3];
print(2 in lst);
print(5 in lst);

print("ell" in "hello");
print("xyz" in "hello");
```

Output:

```
true
false
true
false
true
false
```

### Logical

The `and` and `or` operators use short-circuit evaluation. They return the value
that determined the result, not necessarily `true` or `false`:

```rust
let a = true and false;
print(a);

let b = true or false;
print(b);

let c = false or "fallback";
print(c);
```

Output:

```
false
true
fallback
```

Because of short-circuiting, the right-hand side is not evaluated when the
result is already determined:

```rust
print(false and panic("not reached"));
print(true or panic("not reached"));
```

Output:

```
false
true
```

Truthiness is used by `if`, `while`, `and`, `or`, `!`, `assert`, `.filter()`,
and similar predicates. The falsy values are `None`, `false`, numeric zero,
empty strings, empty lists, empty dictionaries, and the empty tuple `()`. Other
values are truthy.

### Negation

The `!` operator returns the boolean negation of a value's truthiness:

```rust
print(!true);
print(!0);
print(!"");
print(![1]);
```

Output:

```
false
true
true
false
```

### Range

The `..` operator creates a range. The `Range()` built-in provides more control
including custom step values:

```rust
for i in 0..5 {
    print(i);
}
```

Output:

```
0
1
2
3
4
```

Ranges exclude the upper bound. The `Range()` built-in supports a `.step()`
method for custom increments:

```rust
for i in Range(0, 10).step(3) {
    print(i);
}
```

Output:

```
0
3
6
9
```

Negative steps count downward:

```rust
for i in Range(5, 0).step(-1) {
    print(i);
}
```

Output:

```
5
4
3
2
1
```

### Assignment Operators

Compound assignment operators combine an arithmetic operation with assignment.
They modify a variable, struct field, or list element in place:

| Operator | Equivalent to |
|----------|---------------|
| `+=` | `x = x + value` |
| `-=` | `x = x - value` |
| `*=` | `x = x * value` |
| `/=` | `x = x / value` |
| `%=` | `x = x % value` |

```rust
let count = 0;
count += 5;
count -= 2;
count *= 4;
count /= 3;
count %= 3;
print(count);
```

Output:

```
1
```

These operators also work on struct fields and list elements:

```rust
struct Point { x, y }
let p = Point(1, 2);
p.x += 10;
print(p.x);

let lst = [10, 20, 30];
lst[0] += 5;
print(lst[0]);
```

Output:

```
11
15
```

## Control Flow

### If / Else / Else If

Conditionals use `if` and `else` with curly braces. Chains of conditions use
`else if`:

```rust
let x = 10;
if x > 5 {
    print("big");
} else {
    print("small");
}
```

Output:

```
big
```

`else if` flattens conditional chains without nesting:

```rust
let x = 3;
if x == 1 {
    print("one");
} else if x == 2 {
    print("two");
} else if x == 3 {
    print("three");
} else {
    print("other");
}
```

Output:

```
three
```

`if`/`else if`/`else` can also be used as an expression (see
[Block Expressions](#block-expressions)).

### While Loops

```rust
let i = 0;
while i < 3 {
    print(i);
    i = i + 1;
}
```

Output:

```
0
1
2
```

### For Loops

`for` loops iterate over any iterable value — lists, ranges, dictionary
keys, and custom iterators:

```rust
let fruits = ["apple", "banana", "cherry"];
for fruit in fruits {
    print(fruit);
}
```

Output:

```
apple
banana
cherry
```

```rust
for i in 0..5 {
    print(i);
}
```

Output:

```
0
1
2
3
4
```

### Break and Continue

`break` exits a loop early. `continue` skips to the next iteration:

```rust
let i = 0;
while true {
    if i == 2 {
        i = i + 1;
        continue;
    }
    if i == 4 {
        break;
    }
    print(i);
    i = i + 1;
}
```

Output:

```
0
1
3
```

## Functions

### Defining Functions

Functions are defined with the `fn` keyword:

```rust
fn add(a, b) {
    return a + b;
}
print(add(3, 4));
```

Output:

```
7
```

### Implicit Return

The last expression in a function body is its return value when no explicit
`return` statement is used:

```rust
fn double(x) {
    x * 2
}
print(double(5));
```

Output:

```
10
```

### Default Arguments

Parameters can have default values. Default expressions are evaluated each time
the function is called with that argument omitted:

```rust
fn greet(name="World") {
    print("Hello, " + name + "!");
}
greet();
greet("Alice");
```

Output:

```
Hello, World!
Hello, Alice!
```

Required parameters cannot appear after default parameters.

### Keyword Arguments

Arguments can be passed by name, allowing you to skip positional order:

```rust
fn describe(name, age, city) {
    print(name, "is", age, "from", city);
}
describe("Alice", "NYC", age=30);
```

Output:

```
Alice is 30 from NYC
```

Positional arguments cannot appear after keyword arguments. Function calls and
literal lists/tuples allow trailing commas.

### Anonymous Functions

Anonymous functions are created with `fn` without a name. They are commonly used
as callbacks:

```rust
let square = fn(x) { x * x };
print(square(5));
```

Output:

```
25
```

### Closures

Functions capture variables from their enclosing scope, forming closures:

```rust
fn make_counter(start) {
    let count = start;
    fn increment() {
        count = count + 1;
        count
    }
    increment
}

let counter = make_counter(0);
print(counter());
print(counter());
print(counter());
```

Output:

```
1
2
3
```

Closures work with anonymous functions as well:

```rust
fn make_adder(n) {
    let adder = fn(val) { n + val };
    adder
}

let add_5 = make_adder(5);
let add_10 = make_adder(10);
print(add_5(3));
print(add_10(3));
```

Output:

```
8
13
```

## Structs

### Defining Structs

Structs group named fields together. Instances are created by calling the struct
name as a constructor:

```rust
struct Point {
    x,
    y
}

let p = Point(1, 2);
print(p.x, p.y);
```

Output:

```
1 2
```

Structs print with a readable representation:

```rust
struct Color {
    r, g, b
}
let c = Color(255, 128, 0);
print(c);
```

Output:

```
Color(r=255, g=128, b=0)
```

### Methods

Methods are defined inside the struct body and take `self` as the first
parameter:

```rust
struct Point {
    x,
    y

    fn add(self, other) {
        Point(self.x + other.x, self.y + other.y)
    }
}

let a = Point(1, 2);
let b = Point(3, 4);
let c = a.add(b);
print(c.x, c.y);
```

Output:

```
4 6
```

Field values can be updated through `self`:

```rust
struct Point {
    x, y
}

let p = Point(2, 3);
p.x = p.x + 1;
print(p.x);
```

Output:

```
3
```

### Static Methods

Methods without a `self` parameter are static methods, called on the struct
itself rather than an instance:

```rust
struct Point {
    x, y

    fn origin() {
        Point(0, 0)
    }
}

let p = Point.origin();
print(p.x, p.y);
```

Output:

```
0 0
```

### Default Field Values

Struct fields can have default values, including computed expressions:

```rust
struct Config {
    debug = false,
    verbose = false
}

let cfg = Config();
print(cfg.debug, cfg.verbose);

let cfg2 = Config(debug=true);
print(cfg2.debug, cfg2.verbose);
```

Output:

```
false false
true false
```

Required fields cannot appear after fields with default values. Struct
constructors accept positional and keyword arguments, using field names for
keywords.

### Operator Overloading

Structs can override the following operators by defining methods with these
names:

| Method | Operator |
|--------|----------|
| `add` | `+` |
| `sub` | `-` |
| `mul` | `*` |
| `div` | `/` |
| `modulus` | `%` |
| `eq` | `==` |
| `gt` | `>` |
| `gte` | `>=` |
| `lt` | `<` |
| `lte` | `<=` |
| `contains` | `in` |

Each method takes `self` and the right-hand operand as arguments and returns the
result:

```rust
struct Vec2 {
    x, y

    fn add(self, other) {
        Vec2(self.x + other.x, self.y + other.y)
    }

    fn eq(self, other) {
        self.x == other.x and self.y == other.y
    }
}

let a = Vec2(1, 2);
let b = Vec2(3, 4);
let c = a + b;
print(c.x, c.y);
print(a == b);
print(a == Vec2(1, 2));
```

Output:

```
4 6
false
true
```

### Struct Introspection

Every struct definition has a `__name__` attribute. Instances have a `__type__`
attribute that refers back to the struct definition:

```rust
struct Animal {
    name,
    age
}

print(Animal.__name__);

let dog = Animal("Buddy", 5);
print(dog.__type__.__name__);
```

Output:

```
Animal
Animal
```

## String Interpolation

Expressions can be embedded in strings using `\(expr)` syntax:

```rust
let name = "World";
let msg = "Hello, \(name)!";
print(msg);
```

Output:

```
Hello, World!
```

Arbitrary expressions are supported, including function calls and nested
interpolation:

```rust
let result = "2 + 3 = \(2 + 3)";
print(result);
```

Output:

```
2 + 3 = 5
```

```rust
fn greet(name) {
    "hello \(name)"
}
print("You say: '\(greet("Alice"))' to me");
```

Output:

```
You say: 'hello Alice' to me
```

### The `.format` method

Interpolating a value is implemented by calling its `.format` method, so
`"Value: \(x)"` is equivalent to `"Value: " + x.format()`. Every value type
has a default `.format` implementation that produces its standard string
representation.

The interpolation may pass positional and keyword arguments to `.format`. For
example, `"\(value, pretty=true)"` calls `value.format(pretty=true)`. The
default `.format` does not accept any extra arguments (passing some raises an
error), but a type can override `format` to accept them. A struct, for
instance, can override `format` to customize how it is rendered:

```rust
struct Point {
    x,
    y,

    fn format(self, pretty=false) {
        if pretty {
            "Point { x: \(self.x), y: \(self.y) }"
        } else {
            "(\(self.x), \(self.y))"
        }
    }
}

let p = Point(1, 2);
print("\(p)");
print("\(p, pretty=true)");
```

Output:

```
(1, 2)
Point { x: 1, y: 2 }
```

#### Formatting floats

The `.format` method on floats accepts several options (all optional), which
can be supplied as positional or keyword arguments:

- `fill`: the single character used to pad empty space (defaults to `" "`)
- `align`: `"left"`, `"center"`, or `"right"` (defaults to `"right"`)
- `force_sign`: always show the `+`/`-` sign (defaults to `false`)
- `width`: the total width of the formatted string
- `precision`: the number of digits to show after the decimal point
- `notation`: `"e"` or `"E"` to force scientific notation

```rust
print("\(3.14159, precision=2)");
print("\(3.14, width=8, fill="0")");
print("\(3.14, force_sign=true)");
print("\(1234.5, notation="e")");
```

Output:

```
3.14
00003.14
+3.14
1.2345e3
```

## Block Expressions

Curly braces create block expressions. The value of the last expression in a
block becomes the block's value:

```rust
let result = {
    let a = 10;
    let b = 20;
    a + b
};
print(result);
```

Output:

```
30
```

This works with `if`/`else` to create conditional expressions:

```rust
let x = 10;
let label = if x > 5 { "big" } else { "small" };
print(label);
```

Output:

```
big
```

## Statement Terminators

`let` declarations, `return <expr>`, `break`, `continue`, assignments,
compound assignments, and expression statements require semicolons. Function,
struct, `if`, `while`, and `for` declarations/statements do not use a trailing
semicolon after their closing brace.

A final expression in a block may omit the semicolon; that expression becomes
the block's value. Adding a semicolon makes it an expression statement instead.

## Comments

Single-line comments start with `//`:

```rust
// This is a comment
let x = 42;
print(x);
```

Output:

```
42
```

Multi-line comments use `/* */` and can be nested:

```rust
/* This is a
   multi-line comment */
let x = 42;

/* outer /* inner comment */ still a comment */
print(x);
```

Output:

```
42
```

## Custom Iterators

An iterable is any value with an `iter` method. Calling `.iter()` returns an
iterator object, and that iterator exposes a `next` method. `for` loops call
`.iter()` for you and repeatedly call `.next()` until it returns `None`.

Any struct that implements an `iter` method returning an iterator object can be
used in `for` loops. The `next` method should return `None` to signal the end
of iteration:

```rust
struct Counter {
    current,
    max,

    fn iter(self) {
        self
    }

    fn next(self) {
        if self.current >= self.max {
            return None;
        }
        let val = self.current;
        self.current = self.current + 1;
        return val;
    }
}

for i in Counter(0, 3) {
    print(i);
}
```

Output:

```
0
1
2
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `print(args...)` | Prints arguments separated by spaces, followed by a newline |
| `assert(condition)` | Panics if `condition` is falsy |
| `panic(message)` | Immediately halts execution with an error message |
| `dict()` | Creates a new empty dictionary |
| `dict(key=value, ...)` | Creates a dictionary with string keys from keyword names |
| `Range(start, end)` | Creates a range from `start` (inclusive) to `end` (exclusive) |
| `enumerate(iterable)` | Returns an iterable yielding `(index, value)` tuples |
| `filter(iterable)` | Returns a list of truthy values from any iterable |
| `filter(iterable, fn)` | Returns a list of values where `fn(value)` is truthy |
| `average(iterable)` | Returns the arithmetic average, or `0` for an empty iterable |
| `bool(value)` | Converts a value's truthiness to a boolean |
| `str(value)` | Converts a value to a string |
| `int(value)` | Converts a value to an integer (panics on failure) |
| `float(value)` | Converts a value to a float (panics on failure) |
| `try_int(value)` | Converts a value to an integer, returns `None` on failure |
| `try_float(value)` | Converts a value to a float, returns `None` on failure |

Type conversion examples:

```rust
print(int("42"));
print(float("3.14"));
print(str(123));
```

Output:

```
42
3.14
123
```

The `try_` variants return `None` instead of panicking on invalid input:

```rust
let good = try_int("42");
print(good);
let bad = try_int("hello");
print(bad);
```

Output:

```
42
None
```

Generic iterable helpers work with lists, ranges, strings, dictionaries, and
custom iterators:

```rust
for i, value in enumerate(["a", "b"]) {
    print(i, value);
}

print(filter([0, 1, "", "ok"]));
print(average([2, 4, 6]));
print(average([]));
```

Output:

```
0 a
1 b
[1, ok]
4.0
0
```

## Error Handling Philosophy

Shimlang intentionally does not include a `try`/`catch` mechanism. Exceptions
are not part of normal control flow. When an invalid operation occurs — such as
an out-of-bounds access, a failed type conversion, or a violated assertion — it
represents programmer error, not a recoverable condition.

The built-in `panic` function and `assert` halt execution immediately:

```rust
assert(1 == 1);   // passes silently
assert(1 == 2);   // halts with an error

panic("something went wrong");  // always halts
```

For operations that might legitimately fail with user-supplied data, Shimlang
provides `try_` variants that return `None` instead of panicking:

```rust
let n = try_int("not a number");
if n == None {
    print("invalid input");
}
```

Output:

```
invalid input
```

Rather than adding exception handling, the plan is to introduce a
**time-travelling debugger with hot reloading** that automatically opens a
debugger at the offending line during development. This means that when a panic
occurs, the developer can inspect the full program state, step backward through
execution, fix the code, and resume — without restarting the program. GDScript
takes a similar approach by treating errors as signals for the development
environment rather than conditions to catch at runtime.

## Suggested Improvements

The following changes would improve the Shimlang experience for users:

- **Error handling:** The absence of `try`/`catch` is a deliberate design
  choice (see [Error Handling Philosophy](#error-handling-philosophy)). Future
  work in this area focuses on the time-travelling debugger and expanding the
  set of `try_` variants for operations that can fail with user-supplied data.
- **Module / import system:** All code lives in a single file. An import
  system would allow organizing projects into reusable modules.
- **Let-binding destructuring:** Tuples currently destructure only in `for`
  loops. A `let (x, y) = point;` form would make working with multiple return
  values more ergonomic.
- **Variadic arguments (`*args`, `**kwargs`):** Functions cannot currently
  accept a variable number of positional or keyword arguments.
- **Standard library expansion:** Built-in support for file I/O and additional
  collection, path, and text-processing helpers would greatly increase the
  language's utility.
- **REPL improvements:** The interactive REPL could benefit from line editing,
  history, and multi-line input support.
- **Better error messages:** While error messages include source location,
  they could be improved with suggestions, "did you mean?" hints, and
  stack traces for runtime errors in nested function calls.
