use crate::lex::debug_u8s;
use crate::mem::*;
use crate::runtime::*;
#[cfg(feature = "tracy")]
use shm_tracy::*;
use std::any::type_name;

pub(crate) struct ListIterator {
    pub(crate) lst: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for ListIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_list_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to ListIterator.next()".to_string());
                }

                let itr: &mut ListIterator = args.args[0].as_native(interpreter)?;
                let lst = itr.lst.list(interpreter)?;
                if itr.idx >= lst.len() {
                    Ok(ShimValue::StopIteration)
                } else {
                    let result = lst.get(&interpreter.mem, itr.idx as isize)?;
                    itr.idx += 1;

                    Ok(result)
                }
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_list_iter_next)?)
        } else if ident == b"iter" {
            fn shim_list_iterator_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_list_iterator_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.lst]
    }
}


pub(crate) struct Enumerator {
    pub(crate) obj: ShimValue,
}
impl ShimNative for Enumerator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"iter" {
            fn shim_enumerator_iter(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to Enumerator.iter()".to_string());
                }
                let obj = {
                    let enumerator: &mut Enumerator = args.args[0].as_native(interpreter)?;
                    enumerator.obj
                };

                let mut pending_args = ArgBundle::new();
                let inner_iter = match obj.attr_call(b"iter", interpreter, &mut pending_args)? {
                    CallResult::ReturnValue(v) => v,
                    CallResult::PC(pc, captured_scope) => {
                        let mut new_env = Environment::with_scope(captured_scope);
                        interpreter.execute_bytecode_extended(
                            &mut (pc as usize),
                            pending_args,
                            &mut new_env,
                        )?
                    }
                };

                Ok(interpreter
                    .mem
                    .alloc_native(EnumeratorIterator { inner_iter, idx: 0 })?)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_enumerator_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.obj]
    }
}

pub(crate) struct EnumeratorIterator {
    pub(crate) inner_iter: ShimValue,
    pub(crate) idx: i32,
}
impl ShimNative for EnumeratorIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_enumerator_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(
                        "Can't provide positional args to EnumeratorIterator.next()".to_string(),
                    );
                }
                let inner_iter = {
                    let itr: &mut EnumeratorIterator = args.args[0].as_native(interpreter)?;
                    itr.inner_iter
                };

                let mut pending_args = ArgBundle::new();
                let val = match inner_iter.attr_call(b"next", interpreter, &mut pending_args)? {
                    CallResult::ReturnValue(v) => v,
                    CallResult::PC(pc, captured_scope) => {
                        let mut new_env = Environment::with_scope(captured_scope);
                        interpreter.execute_bytecode_extended(
                            &mut (pc as usize),
                            pending_args,
                            &mut new_env,
                        )?
                    }
                };

                if val.is_stop_iteration() {
                    return Ok(ShimValue::StopIteration);
                }

                let itr: &mut EnumeratorIterator = args.args[0].as_native(interpreter)?;
                let idx = ShimValue::Integer(itr.idx);
                itr.idx += 1;

                Ok(interpreter.mem.alloc_tuple(&[idx, val])?)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_enumerator_iter_next)?)
        } else if ident == b"iter" {
            fn shim_enumerator_iterator_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_enumerator_iterator_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.inner_iter]
    }
}

pub(crate) struct StringIterator {
    pub(crate) str_val: ShimValue,
    pub(crate) idx: usize,
}

impl ShimNative for StringIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_str_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to StringIterator.next()"));
                }

                let itr: &mut StringIterator = args.args[0].as_native(interpreter)?;
                let b = {
                    let s = itr.str_val.string(interpreter)?;
                    if itr.idx >= s.len() {
                        return Ok(ShimValue::StopIteration);
                    }
                    s[itr.idx]
                };
                itr.idx += 1;
                interpreter.mem.alloc_str(&[b])
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_str_iter_next)?)
        } else if ident == b"iter" {
            fn shim_str_iterator_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_str_iterator_iter)?)
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>()))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.str_val]
    }
}

pub(crate) struct DictKeysIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictKeysIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_keys_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to DictKeysIterator.next()".to_string());
                }

                let itr: &mut DictKeysIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);

                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].key;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }

                Ok(ShimValue::StopIteration)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_next)?)
        } else if ident == b"iter" {
            fn shim_dict_keys_iter_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct DictValuesIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictValuesIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_values_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to DictValuesIterator.next()".to_string());
                }

                let itr: &mut DictValuesIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);

                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].value;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }

                Ok(ShimValue::StopIteration)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_values_iter_next)?)
        } else if ident == b"iter" {
            fn shim_dict_values_iter_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_values_iter_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct DictItemsIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictItemsIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_items_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to DictItemsIterator.next()".to_string());
                }

                let itr: &mut DictItemsIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);

                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let entry = &entries[itr.idx];
                        let result = interpreter.mem.alloc_tuple(&[entry.key, entry.value])?;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }

                Ok(ShimValue::StopIteration)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_items_iter_next)?)
        } else if ident == b"iter" {
            fn shim_dict_items_iter_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_dict_items_iter_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct RangeNative {
    pub(crate) start: ShimValue,
    pub(crate) end: ShimValue,
}

impl ShimNative for RangeNative {
    fn to_string(&self, interpreter: &mut Interpreter) -> String {
        format!(
            "Range({}, {})",
            self.start.to_string(interpreter),
            self.end.to_string(interpreter)
        )
    }

    fn to_string_mem(&self, mem: &MMU) -> String {
        format!(
            "Range({}, {})",
            self.start.to_string_mem(mem),
            self.end.to_string_mem(mem)
        )
    }

    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"step" {
            fn shim_range_step(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                let step = unpacker.required(b"step")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;

                // Check for zero step
                let is_zero = matches!(step, ShimValue::Integer(0) | ShimValue::Float(0.0));

                if is_zero {
                    return Err("Step cannot be zero".to_string());
                }

                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step,
                };
                Ok(interpreter.mem.alloc_native(iterator)?)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_range_step)?)
        } else if ident == b"iter" {
            fn shim_range_iter(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;
                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step: ShimValue::Integer(1),
                };
                Ok(interpreter.mem.alloc_native(iterator)?)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_range_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.start, self.end]
    }
}

pub(crate) struct RangeIterator {
    pub(crate) current: ShimValue,
    pub(crate) end: ShimValue,
    pub(crate) step: ShimValue,
}

impl ShimNative for RangeIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_range_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to RangeIterator.next()".to_string());
                }

                let itr: &mut RangeIterator = args.args[0].as_native(interpreter)?;

                // Determine if we've reached the end based on step direction
                // For positive steps: iterate while current < end
                // For negative steps: iterate while current > end
                let step_is_positive = match itr.step.gt(interpreter, &ShimValue::Integer(0))? {
                    ShimValue::Bool(b) => b,
                    _ => return Err("Step comparison failed".to_string()),
                };

                let has_more = if step_is_positive {
                    // current < end
                    match itr.current.lt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err("Range comparison failed".to_string()),
                    }
                } else {
                    // current > end
                    match itr.current.gt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err("Range comparison failed".to_string()),
                    }
                };

                if !has_more {
                    Ok(ShimValue::StopIteration)
                } else {
                    let result = itr.current;
                    // current = current + step
                    let mut pending_args = ArgBundle::new();
                    match itr.current.add(interpreter, &itr.step, &mut pending_args)? {
                        CallResult::ReturnValue(new_current) => {
                            itr.current = new_current;
                            Ok(result)
                        }
                        CallResult::PC(pc, captured_scope) => {
                            let mut new_env = Environment::with_scope(captured_scope);
                            let new_current = interpreter.execute_bytecode_extended(
                                &mut (pc as usize),
                                pending_args,
                                &mut new_env,
                            )?;
                            itr.current = new_current;
                            Ok(result)
                        }
                    }
                }
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_range_iter_next)?)
        } else if ident == b"iter" {
            fn shim_range_iterator_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_range_iterator_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.current, self.end, self.step]
    }
}

/// An iterator backed by a user-provided function. Its `.next` simply calls the
/// stored function, and `.iter` returns itself, so it acts as both an iterable
/// and an iterator. This lets users implement iterators without defining a
/// bespoke iterator struct, and lets `Iterator(...)` be returned directly from a
/// struct's `iter` method.
pub(crate) struct FuncIterator {
    pub(crate) func: ShimValue,
}

impl ShimNative for FuncIterator {
    fn to_string(&self, _interpreter: &mut Interpreter) -> String {
        "Iterator(...)".to_string()
    }

    fn to_string_mem(&self, _mem: &MMU) -> String {
        "Iterator(...)".to_string()
    }

    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_func_iterator_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(
                        "Can't provide positional args to FuncIterator.next()".to_string()
                    );
                }

                let func = {
                    let itr: &FuncIterator = args.args[0].as_native(interpreter)?;
                    itr.func
                };

                let mut call_args = ArgBundle::new();
                match func.call(interpreter, &mut call_args)? {
                    CallResult::ReturnValue(val) => Ok(val),
                    CallResult::PC(pc, captured_scope) => {
                        let mut new_env = Environment::with_scope(captured_scope);
                        interpreter.execute_bytecode_extended(
                            &mut (pc as usize),
                            call_args,
                            &mut new_env,
                        )
                    }
                }
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_func_iterator_next)?)
        } else if ident == b"iter" {
            fn shim_func_iterator_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_func_iterator_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.func]
    }
}

const fn generate_size_table() -> [u32; 256] {
    let mut table = [0; 256];

    let mut i = 0;

    while i < 256 {
        table[i] = match i {
            0 => 0,
            1 => 4,
            2 => 16,
            3 => 32,
            // Multiply 1.5 the previous
            4 => 48,
            5 => 72,
            6 => 108,
            7 => 162,
            8 => 243,
            9 => 364,
            10 => 546,
            11 => 819,
            12 => 1228,
            13 => 1842,
            // Multiply x1.2 the previous
            14 => 2210,
            15 => 2652,
            16 => 3182,
            17 => 3818,
            18 => 4581,
            19 => 5497,
            20 => 6596,
            21 => 7915,
            22 => 9498,
            23 => 11397,
            24 => 13676,
            25 => 16411,
            26 => 19693,
            27 => 23631,
            28 => 28357,
            29 => 34028,
            30 => 40833,
            31 => 48999,
            32 => 58798,
            33 => 70557,
            34 => 84668,
            35 => 101601,
            36 => 121921,
            37 => 146305,
            38 => 175566,
            39 => 210679,
            40 => 252814,
            41 => 303376,
            42 => 364051,
            43 => 436861,
            44 => 524233,
            45 => 629079,
            46 => 754894,
            47 => 905872,
            48 => 1087046,
            49 => 1304455,
            50 => 1565346,
            51 => 1878415,
            52 => 2254098,
            53 => 2704917,
            54 => 3245900,
            55 => 3895080,
            56 => 4674096,
            57 => 5608915,
            58 => 6730698,
            59 => 8076837,
            60 => 9692204,
            61 => 11630644,
            62 => 13956772,
            63 => 16748126,
            _ => MAX_U24,
        };
        i += 1;
    }
    table
}

static LIST_CAPACITY_LUT: [u32; 256] = generate_size_table();

#[derive(Debug, Clone, Copy)]
pub struct DictEntry {
    pub hash: u64,
    pub key: ShimValue,
    pub value: ShimValue,
}

impl DictEntry {
    pub(crate) fn is_valid(&self) -> bool {
        self.hash != 0 && !self.key.is_uninitialized() && !self.value.is_uninitialized()
    }

    fn invalidate(&mut self) {
        self.hash = 0;
        self.key = ShimValue::Uninitialized;
        self.value = ShimValue::Uninitialized;
    }
}

// Minimum non-zero size_pow for ShimDict. When the dict grows from empty,
// it starts with this size_pow value (2^3 = 8 index slots, capacity of ~5 entries).
const MIN_NON_ZERO_SIZE_POW: u8 = 3;

#[derive(Debug)]
pub struct ShimDict {
    // Size of the index array, always a power of 2
    pub(crate) size_pow: u8,

    // These could be u24, but are u32 to keep things simple

    // Number of valid entries + tombstoned entries
    pub(crate) entry_count: u32,
    // Non-tombstoned entries
    pub(crate) used: u32,

    // Memory position of the dict data
    pub(crate) indices: u24,
    pub(crate) entries: u24,
}

enum DictSlot<'a> {
    Occupied(usize, &'a mut DictEntry),
    // If it's Unoccupied, this is the idx in the indices array
    UnoccupiedU8(u32, usize),
    UnoccupiedU16(u32, usize),
    UnoccupiedU32(u32, usize),
}

#[derive(Debug)]
enum TypedIndices {
    Zero,
    U8(&'static mut [u8]),
    U16(&'static mut [u16]),
    U32(&'static mut [u32]),
}

impl TypedIndices {
    fn get(&self, index: usize) -> usize {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => data[index] as usize,
            Self::U16(data) => data[index] as usize,
            Self::U32(data) => data[index] as usize,
        }
    }

    fn set(&mut self, index: usize, value: usize) {
        match self {
            Self::Zero => panic!("Can't IndexMut empty TypedIndices"),
            Self::U8(data) => data[index] = value as u8,
            Self::U16(data) => data[index] = value as u16,
            Self::U32(data) => data[index] = value as u32,
        }
    }

    fn is_unset(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => data[index] == u8::MAX,
            Self::U16(data) => data[index] == u16::MAX,
            Self::U32(data) => data[index] == u32::MAX,
        }
    }

    fn is_tombstone(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => data[index] == u8::MAX - 1,
            Self::U16(data) => data[index] == u16::MAX - 1,
            Self::U32(data) => data[index] == u32::MAX - 1,
        }
    }

    fn set_tombstone(&mut self, index: usize) {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => data[index] = u8::MAX - 1,
            Self::U16(data) => data[index] = u16::MAX - 1,
            Self::U32(data) => data[index] = u32::MAX - 1,
        }
    }
}

impl ShimDict {
    pub(crate) fn new() -> Self {
        Self {
            size_pow: 0,
            used: 0,
            entry_count: 0,
            indices: 0.into(),
            entries: 0.into(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.used as usize
    }

    pub(crate) fn get(
        &self,
        interpreter: &mut Interpreter,
        key: ShimValue,
    ) -> Result<ShimValue, String> {
        // Check if dict is empty
        if self.size_pow == 0 {
            return Err(format!(
                "Key {} not in dict",
                key.to_string_mem(&interpreter.mem)
            ));
        }

        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => Ok(entry.value),
            DictSlot::UnoccupiedU8(..) => Err(format!(
                "Key {} not in dict",
                key.to_string_mem(&interpreter.mem)
            )),
            _ => todo!(),
        }
    }

    fn print_entries(&self, interpreter: &Interpreter) {
        eprintln!("Entries");
        let _entries: &[DictEntry] = unsafe {
            let u64_slice = &interpreter.mem.mem()[usize::from(self.entries)
                ..usize::from(self.entries) + 3 * (self.entry_count as usize)];
            std::slice::from_raw_parts(u64_slice.as_ptr() as *const DictEntry, u64_slice.len() / 3)
        };
    }

    fn expand_capacity(&mut self, interpreter: &mut Interpreter) -> Result<(), String> {
        let _zone = zone_scoped!("ShimDict::expand_capacity");
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = if old_size == 0 {
            MIN_NON_ZERO_SIZE_POW
        } else {
            self.size_pow + 1
        };

        self.clear_and_alloc_indices(interpreter, old_size)?;
        self.realloc_entries(interpreter, old_capacity)?;
        Ok(())
    }

    fn realloc_entries(&mut self, interpreter: &mut Interpreter, old_capacity: usize) -> Result<(), String> {
        let old_entries_word = self.entries;
        let old_entries = self.entries_array(interpreter);

        let free_word_count: u24 = (old_capacity * 3).into();
        let alloc_word_count: u24 = (self.capacity() * 3).into();
        self.entries = alloc!(interpreter.mem, alloc_word_count, "Dict entry array")?;

        let new_entries = self.entries_mut(interpreter);

        let mut write_idx = 0;
        for &entry in old_entries {
            if entry.is_valid() {
                new_entries[write_idx] = entry;
                new_entries[write_idx].is_valid();
                write_idx += 1;
            }
        }
        // This should be equal to or lower than the previous entry_count since
        // it will remove tombstones
        self.entry_count = write_idx as u32;

        let new_entries = self.entries_array(interpreter);
        let mut indices = self.typed_indices(interpreter);
        for (entry_idx, entry) in new_entries.iter().enumerate() {
            let index_idx = self.probe_entry_realloc(interpreter, entry.hash as u32);
            indices.set(index_idx, entry_idx);
        }

        interpreter.mem.free(old_entries_word, free_word_count);
        Ok(())
    }

    pub fn indices_stride_bytes(&self, size: usize) -> usize {
        if size == 0 {
            0
        } else if size <= (u8::MAX as usize) + 1 {
            1
        } else if size <= (u16::MAX as usize) + 1 {
            2
        } else {
            4
        }
    }

    fn typed_indices(&self, interpreter: &Interpreter) -> TypedIndices {
        match self.index_size() {
            0 => TypedIndices::Zero,
            x if x <= (u8::MAX as usize) + 1 => {
                TypedIndices::U8(self.indicies_mut::<u8>(interpreter))
            }
            x if x <= (u16::MAX as usize) + 1 => {
                TypedIndices::U16(self.indicies_mut::<u16>(interpreter))
            }
            _ => TypedIndices::U32(self.indicies_mut::<u32>(interpreter)),
            // On wasm32 (32-bit usize), all values fit in u32 so the catch-all above covers everything.
            // On 64-bit, the memory system is bounded by MAX_U24 * 8 bytes, so index_size() can never
            // exceed u32::MAX + 1 in practice.
        }
    }

    /**
     * Clear the indices array with current size
     */
    fn clear_and_alloc_indices(&mut self, interpreter: &mut Interpreter, old_size: usize) -> Result<(), String> {
        let new_size = self.index_size();
        let free_word_count: u24 = if old_size == 0 {
            0.into()
        } else {
            old_size
                .div_ceil(8 / self.indices_stride_bytes(old_size))
                .into()
        };
        let alloc_word_count: u24 = if new_size == 0 {
            0.into()
        } else {
            new_size
                .div_ceil(8 / self.indices_stride_bytes(new_size))
                .into()
        };

        interpreter.mem.free(self.indices, free_word_count);
        self.indices = alloc!(interpreter.mem, alloc_word_count, "Dict index array")?;

        match self.typed_indices(interpreter) {
            TypedIndices::Zero => (),
            TypedIndices::U8(indices) => {
                for x in indices.iter_mut() {
                    *x = u8::MAX;
                }
            }
            TypedIndices::U16(indices) => {
                for x in indices.iter_mut() {
                    *x = u16::MAX;
                }
            }
            TypedIndices::U32(indices) => {
                for x in indices.iter_mut() {
                    *x = u32::MAX;
                }
            }
        }
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        Self::capacity_for_size_pow(self.size_pow)
    }

    fn capacity_for_size_pow(size_pow: u8) -> usize {
        if size_pow == 0 {
            0
        } else {
            let index_size = 1 << size_pow;
            ((index_size * 2) / 3) as usize
        }
    }

    fn index_size(&self) -> usize {
        if self.size_pow == 0 {
            0
        } else {
            (1 << self.size_pow) as usize
        }
    }

    fn mask(&self) -> usize {
        self.index_size() - 1
    }

    fn probe_entry_realloc(&self, interpreter: &Interpreter, longhash: u32) -> usize {
        let mask = self.mask();

        let hash: usize = (longhash as usize) & mask;
        let mut idx = hash & mask;
        match self.typed_indices(interpreter) {
            TypedIndices::Zero => panic!("Can't probe empty dict"),
            TypedIndices::U8(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u8::MAX {
                        return idx;
                    } else if indices[idx] == u8::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            }
            TypedIndices::U16(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u16::MAX {
                        return idx;
                    } else if indices[idx] == u16::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            }
            TypedIndices::U32(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u32::MAX {
                        return idx;
                    } else if indices[idx] == u32::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            }
        }

        panic!("Probe entry realloc failed probing!");
    }

    fn probe(&self, interpreter: &mut Interpreter, key: ShimValue) -> Result<DictSlot<'_>, String> {
        let longhash = key.hash(interpreter)? as usize;
        let mask = self.mask();

        let mut idx = longhash & mask;

        let mut freeslot = None;

        let indices = self.typed_indices(interpreter);
        // Linear probe for now
        for _ in 0..self.index_size() {
            if indices.is_unset(idx) {
                if freeslot.is_none() {
                    freeslot = Some(idx);
                }
                break;
            } else if indices.is_tombstone(idx) {
                if freeslot.is_none() {
                    freeslot = Some(idx);
                }
            } else {
                // Hash matches, let's check the entry and see if the key matches
                let entry_idx = indices.get(idx);
                let entry = self.get_entry_mut(interpreter, entry_idx);
                if key.dict_key_equal(interpreter, &entry.key)? {
                    return Ok(DictSlot::Occupied(idx, entry));
                }
                // Otherwise continue probing
            }
            idx = (idx + 1) & mask;
        }
        let idx = match freeslot {
            Some(idx) => idx,
            None => {
                eprintln!("{self:#?}");
                eprintln!("Capacity: {:#?}  Mask: {}", self.capacity(), mask);
                panic!("Could not find free slot");
            }
        };
        match indices {
            TypedIndices::Zero => panic!("probing nothing"),
            TypedIndices::U8(_) => Ok(DictSlot::UnoccupiedU8(longhash as u32, idx as usize)),
            TypedIndices::U16(_) => Ok(DictSlot::UnoccupiedU16(longhash as u32, idx as usize)),
            TypedIndices::U32(_) => Ok(DictSlot::UnoccupiedU32(longhash as u32, idx as usize)),
        }
    }

    pub fn set(
        &mut self,
        interpreter: &mut Interpreter,
        key: ShimValue,
        val: ShimValue,
    ) -> Result<(), String> {
        if self.entry_count as usize == self.capacity() {
            self.expand_capacity(interpreter)?;
        }

        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => {
                entry.key = key;
                entry.value = val;
            }
            DictSlot::UnoccupiedU8(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u8>(interpreter)[idx] = entry_idx as u8;
                self.entries_mut(interpreter)[entry_idx].is_valid();
                self.entries_array(interpreter)[entry_idx].is_valid();
                self.used += 1;
            }
            DictSlot::UnoccupiedU16(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u16>(interpreter)[idx] = entry_idx as u16;
                self.used += 1;
            }
            DictSlot::UnoccupiedU32(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u32>(interpreter)[idx] = entry_idx as u32;
                self.used += 1;
            }
        }

        Ok(())
    }

    pub(crate) fn pop(
        &mut self,
        interpreter: &mut Interpreter,
        key: ShimValue,
        default: Option<ShimValue>,
    ) -> Result<ShimValue, String> {
        match self.probe(interpreter, key) {
            Ok(DictSlot::Occupied(indices_idx, entry)) => {
                let value = entry.value;
                entry.hash = 0;
                entry.key = ShimValue::Uninitialized;
                entry.value = ShimValue::Uninitialized;

                let mut indices = self.typed_indices(interpreter);
                indices.set_tombstone(indices_idx);

                // We don't decrement the entry_count since that entry still exists
                self.used -= 1;

                Ok(value)
            }
            Ok(_) => {
                if let Some(default) = default {
                    Ok(default)
                } else {
                    Err(format!("Key {key:?} not found in dict"))
                }
            }
            _ => todo!(),
        }
    }

    fn indicies_mut<T>(&self, interpreter: &Interpreter) -> &'static mut [T] {
        let stride = std::mem::size_of::<T>();
        let size = 1 << self.size_pow;
        let start = usize::from(self.indices);
        let len = size / stride;
        let u64_slice = &interpreter.mem.mem()[start..start + len];
        unsafe {
            std::slice::from_raw_parts_mut(u64_slice.as_ptr() as *mut T, u64_slice.len() * stride)
        }
    }

    /**
     * Return the valid part of the entries array
     */
    pub fn entries_array(&self, interpreter: &Interpreter) -> &'static [DictEntry] {
        unsafe {
            let u64_slice = &interpreter.mem.mem()[usize::from(self.entries)
                ..usize::from(self.entries) + 3 * (self.entry_count as usize)];
            std::slice::from_raw_parts(u64_slice.as_ptr() as *const DictEntry, u64_slice.len() / 3)
        }
    }

    /**
     * Return the entire capacity of the entries table
     */
    fn entries_mut(&self, interpreter: &mut Interpreter) -> &'static mut [DictEntry] {
        unsafe {
            let u64_slice = interpreter.mem.mem_mut(usize::from(self.entries), 3 * self.capacity());
            std::slice::from_raw_parts_mut(
                u64_slice.as_mut_ptr() as *mut DictEntry,
                u64_slice.len() / 3,
            )
        }
    }

    fn get_entry(&self, interpreter: &Interpreter, idx: usize) -> &DictEntry {
        unsafe { &*(interpreter.mem.mem()[usize::from(self.entries) + 3 * idx..].as_ptr() as *const DictEntry) }
    }

    #[allow(clippy::mut_from_ref)]
    fn get_entry_mut(&self, interpreter: &mut Interpreter, idx: usize) -> &mut DictEntry {
        unsafe {
            let start = usize::from(self.entries) + 3 * idx;
            &mut *(interpreter.mem.mem_mut(start, 3).as_mut_ptr() as *mut DictEntry)
        }
    }

    fn set_entry(
        &mut self,
        interpreter: &mut Interpreter,
        hash: u32,
        key: ShimValue,
        val: ShimValue,
    ) -> usize {
        let entry = self.get_entry_mut(interpreter, self.entry_count as usize);
        entry.hash = hash as u64;
        entry.key = key;
        entry.value = val;

        let entry_idx = self.entry_count;
        self.entry_count += 1;
        entry_idx as usize
    }

    pub(crate) fn shrink_to_fit(&mut self, interpreter: &mut Interpreter) -> Result<(), String> {
        if self.used == 0 {
            // Empty dict - reset to minimal size
            let old_size = self.index_size();
            let old_capacity = self.capacity();

            if old_size == 0 {
                return Ok(()); // Already minimal
            }

            self.size_pow = 0;
            self.clear_and_alloc_indices(interpreter, old_size)?;

            // Free the old entries
            let free_word_count: u24 = (old_capacity * 3).into();
            interpreter.mem.free(self.entries, free_word_count);
            self.entries = 0.into();
            self.entry_count = 0;
            return Ok(());
        }

        // Calculate the optimal size_pow for the current number of used entries
        // We want capacity to be at least used, and index_size = capacity * 3 / 2
        // Since index_size must be a power of 2, we find the smallest power of 2
        // such that (2^size_pow * 2 / 3) >= used
        let min_capacity = self.used as usize;
        // Start with MIN_NON_ZERO_SIZE_POW, which matches expand_capacity's initial size
        let mut optimal_size_pow = MIN_NON_ZERO_SIZE_POW;

        // Upper bound of 31 prevents undefined behavior from 1 << 32 and ensures
        // we stay within u32 limits for entry_count/used fields.
        // Loop condition is <= 31 to allow checking if size_pow=31 is sufficient.
        while optimal_size_pow <= 31 {
            let test_capacity = Self::capacity_for_size_pow(optimal_size_pow);
            if test_capacity >= min_capacity {
                break;
            }
            optimal_size_pow += 1;
        }

        // If the optimal size is the same or larger than current, no need to shrink
        if optimal_size_pow >= self.size_pow {
            return Ok(());
        }

        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = optimal_size_pow;

        self.clear_and_alloc_indices(interpreter, old_size)?;
        self.realloc_entries(interpreter, old_capacity)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ShimList {
    // The memory is limited to u24, so we know there can't be more than this
    // number of values
    pub len: u24,
    // We don't really need any more than 64 distinct capacities
    pub capacity_lut: u8,
    // Add 1 byte of padding so that ShimList is 8 bytes
    _pad: u8,
    // Memory position of the list data
    pub(crate) data: u24,
}

const _: () = {
    assert!(std::mem::size_of::<ShimList>() == 8);
};

impl Default for ShimList {
    fn default() -> Self {
        Self::new()
    }
}

impl ShimList {
    pub fn new() -> Self {
        Self {
            len: 0.into(),
            capacity_lut: 0,
            _pad: 0,
            data: 0.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.len.into()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        LIST_CAPACITY_LUT[self.capacity_lut as usize] as usize
    }

    pub fn wrap_idx(&self, idx: isize) -> Result<usize, String> {
        if idx >= self.len() as isize {
            return Err(format!("Index {idx} is out of bounds"));
        }

        Ok(if idx < 0 {
            let updated_idx = self.len() as isize + idx;
            if updated_idx < 0 {
                return Err(format!("Index {idx} is out of bounds"));
            } else {
                updated_idx as usize
            }
        } else {
            idx as usize
        })
    }

    pub fn raw_data<'a>(&self, mem: &'a MMU) -> &'a [u64] {
        &mem.mem()[usize::from(self.data)..usize::from(self.data + self.len)]
    }

    pub fn get(&self, mem: &MMU, idx: isize) -> Result<ShimValue, String> {
        let idx = self.wrap_idx(idx)?;
        unsafe { Ok(ShimValue::from_u64(mem.mem()[usize::from(self.data) + idx])) }
    }

    pub fn set(&self, mem: &mut MMU, idx: isize, value: ShimValue) -> Result<(), String> {
        let idx = self.wrap_idx(idx)?;
        mem.mem_mut(usize::from(self.data) + idx, 1)[0] = value.to_u64();
        Ok(())
    }

    pub fn push(&mut self, mem: &mut MMU, val: ShimValue) -> Result<(), String> {
        if self.len() == self.capacity() {
            let old_capacity = self.capacity();
            self.capacity_lut += 1;
            let new_capacity = self.capacity();

            let old_data = usize::from(self.data);
            let word_count: u24 = new_capacity.into();
            self.data = alloc!(mem, word_count, "List data")?;

            let new_data = usize::from(self.data);

            for idx in 0..self.len() {
                let val = mem.mem()[old_data + idx];
                mem.mem_mut(new_data + idx, 1)[0] = val;
            }

            mem.free(old_data.into(), old_capacity.into());
        }

        mem.mem_mut(usize::from(self.data) + self.len(), 1)[0] = val.to_u64();
        self.len = (usize::from(self.len) + 1).into();
        Ok(())
    }
}
const _: () = {
    assert!(std::mem::size_of::<ShimList>() == 8);
};

pub(crate) fn shim_dict(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    if !args.args.is_empty() {
        return Err("Can't provide positional args to dict()".to_string());
    }

    let retval = interpreter.mem.alloc_dict()?;
    let dict = retval.dict_mut(interpreter)?;

    for (key, val) in args.kwargs.clone().into_iter() {
        let key = interpreter.mem.alloc_str(&key)?;
        dict.set(interpreter, key, val)?;
    }

    Ok(retval)
}

pub(crate) fn shim_enumerate(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(Enumerator { obj })?)
}

pub(crate) fn shim_average(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    let mut args = ArgBundle::new();
    let inner_iter = match obj.attr_call(b"iter", interpreter, &mut args)? {
        CallResult::ReturnValue(v) => v,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(
                &mut (pc as usize),
                args,
                &mut new_env,
            )?
        }
    };

    let mut acc: Option<ShimValue> = None;
    let mut count = 0;
    loop {
        let mut args = ArgBundle::new();
        let val = match inner_iter.attr_call(b"next", interpreter, &mut args)? {
            CallResult::ReturnValue(v) => v,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    &mut new_env,
                )?
            }
        };
        if val.is_stop_iteration() {
            break;
        }
        acc = match acc {
            None => Some(val),
            Some(v) => {
                let mut args = ArgBundle::new();
                Some(
                    match v.add(interpreter, &val, &mut args)? {
                        CallResult::ReturnValue(v) => v,
                        CallResult::PC(pc, captured_scope) => {
                            let mut new_env = Environment::with_scope(captured_scope);
                            interpreter.execute_bytecode_extended(
                                &mut (pc as usize),
                                args,
                                &mut new_env,
                            )?
                        }
                    }
                )
            }
        };
        count += 1;
    }

    match acc {
        Some(total) => total.div(interpreter, &ShimValue::Integer(count)),
        None => Ok(ShimValue::Integer(0)),
    }
}

pub(crate) fn shim_filter(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let predicate = match unpacker.optional(b"predicate") {
        Some(v) => Some(v),
        None => unpacker.optional(b"key"),
    };
    unpacker.end()?;

    filter_iterable(interpreter, obj, predicate)
}

pub(crate) fn shim_map(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let func = unpacker.required(b"func")?;
    unpacker.end()?;

    map_iterable(interpreter, obj, func)
}

fn map_iterable(
    interpreter: &mut Interpreter,
    obj: ShimValue,
    func: ShimValue,
) -> Result<ShimValue, String> {
    let new_lst_val = interpreter.mem.alloc_list()?;

    let mut iter_args = ArgBundle::new();
    let iterator = match obj.attr_call(b"iter", interpreter, &mut iter_args)? {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(&mut (pc as usize), iter_args, &mut new_env)?
        }
    };

    loop {
        let mut next_args = ArgBundle::new();
        let input = match iterator.attr_call(b"next", interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(&mut (pc as usize), next_args, &mut new_env)?
            }
        };

        if input.is_stop_iteration() {
            break;
        }

        let mut args = ArgBundle::new();
        args.args.push(input);
        let output = match func.call(interpreter, &mut args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);

                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    &mut new_env,
                )?
            }
        };

        let new_lst = new_lst_val.list_mut(interpreter)?;
        new_lst.push(&mut interpreter.mem, output)?;
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_iterator(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let func = unpacker.required(b"func")?;
    unpacker.end()?;

    let iterator = FuncIterator { func };
    Ok(interpreter.mem.alloc_native(iterator)?)
}

pub(crate) fn shim_range(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let start = unpacker.required(b"start")?;
    let end = unpacker.required(b"end")?;
    unpacker.end()?;

    let range = RangeNative {
        start,
        end,
    };
    Ok(interpreter.mem.alloc_native(range)?)
}

pub(crate) fn shim_print(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let _zone = zone_scoped!("shim_print");

    // `sep` separates the arguments (default a single space) and `end` is
    // printed after the last argument (default a newline). Any other keyword
    // argument is an error.
    let mut sep = " ".to_string();
    let mut end = "\n".to_string();
    for (ident, val) in args.kwargs.iter() {
        match ident.as_slice() {
            b"sep" => sep = val.to_string(interpreter),
            b"end" => end = val.to_string(interpreter),
            other => {
                return Err(format!(
                    "print got an unexpected keyword argument '{}'",
                    debug_u8s(other)
                ))
            }
        }
    }

    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            print!("{}", sep);
        }
        print!("{}", arg.to_string(interpreter));
    }

    print!("{}", end);
    Ok(ShimValue::None)
}

pub(crate) fn shim_assert(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    if !args.kwargs.is_empty() {
        return Err("Assert doesn't take keyword arguments".to_string());
    }
    if args.len() > 2 {
        return Err(format!(
            "assert takes at most two arguments (condition, message), got {}",
            args.len()
        ));
    }
    if args.len() == 0 {
        return Ok(ShimValue::None);
    }

    let condition = args.args[0];
    if !condition.is_truthy(interpreter)? {
        let msg = if args.len() > 1 {
            args.args[1].to_string(interpreter)
        } else {
            format!("Assert Failed: {} is not truthy", condition.to_string(interpreter))
        };
        Err(msg)
    } else {
        Ok(ShimValue::None)
    }
}

pub(crate) fn shim_panic(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut out = String::new();
    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            out.push(' ');
        }
        out.push_str(&arg.to_string(interpreter));
    }

    out.push('\n');
    Err(out)
}

//enum ShimSortKey {
//    Bytes(&[u8]),
//    Int(i32),
//    Float(f32),
//}

pub(crate) fn shim_list_sort(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a vector of (index, value, sort_key) tuples to maintain stability
    let mut items_with_keys: Vec<(usize, ShimValue, ShimValue)> = Vec::new();

    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;

        let sort_key = if let Some(key) = key {
            let mut args = ArgBundle::new();
            args.args.push(item);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => val,
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    interpreter.execute_bytecode_extended(&mut (pc as usize), args, &mut new_env)?
                }
            }
        } else {
            item
        };

        items_with_keys.push((idx, item, sort_key));
    }

    // Perform stable sort by comparing sort keys
    items_with_keys.sort_by(|a, b| {
        let (idx_a, _, key_a) = a;
        let (idx_b, _, key_b) = b;

        // Try to compare the keys
        match compare_values(interpreter, key_a, key_b) {
            Ok(ordering) => ordering,
            Err(_) => {
                // If comparison fails, maintain original order (stability)
                idx_a.cmp(idx_b)
            }
        }
    });

    // Mutate the list in place
    let lst_mut = obj.list_mut(interpreter)?;
    for (idx, (_, item, _)) in items_with_keys.iter().enumerate() {
        lst_mut.set(&mut interpreter.mem, idx as isize, *item)?;
    }

    Ok(ShimValue::None)
}

// Helper function to compare two ShimValues for sorting/ordering purposes.
// This function returns an Ordering to determine relative position in a sorted sequence.
// For equality checks, use ShimValue::equal_inner instead.
pub(crate) fn compare_values(
    interpreter: &mut Interpreter,
    a: &ShimValue,
    b: &ShimValue,
) -> Result<std::cmp::Ordering, String> {
    use std::cmp::Ordering;

    match (a, b) {
        (ShimValue::Integer(x), ShimValue::Integer(y)) => Ok(x.cmp(y)),
        (ShimValue::Float(x), ShimValue::Float(y)) => {
            // Handle NaN comparison by treating NaN as equal to itself
            if x.is_nan() && y.is_nan() {
                Ok(Ordering::Equal)
            } else if x.is_nan() {
                Ok(Ordering::Greater)
            } else if y.is_nan() || x < y {
                Ok(Ordering::Less)
            } else if x > y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        }
        (ShimValue::Integer(x), ShimValue::Float(y)) => {
            let x_f = *x as f32;
            if x_f < *y {
                Ok(Ordering::Less)
            } else if x_f > *y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        }
        (ShimValue::Float(x), ShimValue::Integer(y)) => {
            let y_f = *y as f32;
            if *x < y_f {
                Ok(Ordering::Less)
            } else if *x > y_f {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        }
        (ShimValue::String(..), ShimValue::String(..)) => {
            let str_a = a.string(interpreter)?;
            let str_b = b.string(interpreter)?;
            Ok(str_a.cmp(str_b))
        }
        (ShimValue::Bool(x), ShimValue::Bool(y)) => Ok(x.cmp(y)),
        (ShimValue::None, ShimValue::None) => Ok(Ordering::Equal),
        (ShimValue::List(_), ShimValue::List(_)) => {
            // Compare lists lexicographically
            let lst_a = a.list(interpreter)?;
            let lst_b = b.list(interpreter)?;

            let min_len = std::cmp::min(lst_a.len(), lst_b.len());
            for i in 0..min_len {
                let item_a = lst_a.get(&interpreter.mem, i as isize)?;
                let item_b = lst_b.get(&interpreter.mem, i as isize)?;
                match compare_values(interpreter, &item_a, &item_b)? {
                    Ordering::Equal => continue,
                    other => return Ok(other),
                }
            }
            Ok(lst_a.len().cmp(&lst_b.len()))
        }
        (ShimValue::Tuple(len_a, pos_a), ShimValue::Tuple(len_b, pos_b)) => {
            let len_a = usize::from(*len_a);
            let len_b = usize::from(*len_b);
            let pos_a = usize::from(*pos_a);
            let pos_b = usize::from(*pos_b);

            let min_len = len_a.min(len_b);
            for idx in 0..min_len {
                let item_a = unsafe { ShimValue::from_u64(interpreter.mem.mem()[pos_a+idx]) };
                let item_b = unsafe { ShimValue::from_u64(interpreter.mem.mem()[pos_b+idx]) };
                match compare_values(interpreter, &item_a, &item_b)? {
                    Ordering::Equal => continue,
                    other => return Ok(other),
                }
            }
            Ok(len_a.cmp(&len_b))
        }
        (ShimValue::Struct(..), _) => {
            // Try struct method overrides for comparison operators
            if let Some(gt_result) = a.try_struct_override(interpreter, b"gt", b) {
                let gt_val: ShimValue = gt_result?;
                if gt_val.is_truthy(interpreter)? {
                    return Ok(Ordering::Greater);
                }
            }
            if let Some(lt_result) = a.try_struct_override(interpreter, b"lt", b) {
                let lt_val: ShimValue = lt_result?;
                if lt_val.is_truthy(interpreter)? {
                    return Ok(Ordering::Less);
                }
            }
            if let Some(eq_result) = a.try_struct_override(interpreter, b"eq", b) {
                let eq_val: ShimValue = eq_result?;
                if eq_val.is_truthy(interpreter)? {
                    return Ok(Ordering::Equal);
                }
            }
            Err(format!(
                "Cannot compare {} and {}",
                a.to_string_mem(&interpreter.mem),
                b.to_string_mem(&interpreter.mem)
            ))
        }
        _ => Err(format!(
            "Cannot compare {} and {}",
            a.to_string_mem(&interpreter.mem),
            b.to_string_mem(&interpreter.mem)
        )),
    }
}

fn filter_iterable(
    interpreter: &mut Interpreter,
    obj: ShimValue,
    predicate: Option<ShimValue>,
) -> Result<ShimValue, String> {
    let new_lst_val = interpreter.mem.alloc_list()?;

    let mut iter_args = ArgBundle::new();
    let iterator = match obj.attr_call(b"iter", interpreter, &mut iter_args)? {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(&mut (pc as usize), iter_args, &mut new_env)?
        }
    };

    loop {
        let mut next_args = ArgBundle::new();
        let input = match iterator.attr_call(b"next", interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(&mut (pc as usize), next_args, &mut new_env)?
            }
        };

        if input.is_stop_iteration() {
            break;
        }

        let result = if let Some(key) = predicate {
            let mut args = ArgBundle::new();
            args.args.push(input);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => val,
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    
                    interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        &mut new_env,
                    )?
                }
            }
        } else {
            input
        };
        if result.is_truthy(interpreter)? {
            let new_lst = new_lst_val.list_mut(interpreter)?;
            new_lst.push(&mut interpreter.mem, input)?;
        }
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_list_join(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let s = unpacker.required(b"s")?;
    unpacker.end()?;

    let args = ArgBundle {
        args: vec![s, obj],
        kwargs: Vec::new(),
    };

    shim_str_join(interpreter, &args)
}

pub(crate) fn shim_list_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(lst.len() as i32))
}

pub(crate) fn shim_list_append(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    lst.push(&mut interpreter.mem, item)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_iter(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter
        .mem
        .alloc_native(ListIterator { lst: obj, idx: 0 })?)
}

pub(crate) fn shim_str_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(StringIterator { str_val: obj, idx: 0 })?)
}

pub(crate) fn shim_list_clear(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    lst.len = 0.into();

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_extend(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let iterable = unpacker.required(b"iterable")?;
    unpacker.end()?;

    // Get the iterator for the iterable
    let mut iter_args = ArgBundle::new();
    let iterator = iterable
        .get_attr(interpreter, b"iter")?
        .call(interpreter, &mut iter_args)?;
    let iterator = match iterator {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(&mut (pc as usize), iter_args, &mut new_env)?
        }
    };

    // Get the next method
    let next_method = iterator.get_attr(interpreter, b"next")?;

    // Iterate and append each item
    loop {
        let mut next_args = ArgBundle::new();

        let result = match next_method.call(interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    next_args,
                    &mut new_env,
                )?
            }
        };

        // Break if we get StopIteration (end of iteration)
        if result.is_stop_iteration() {
            break;
        }

        // Append the item to the list
        let lst = obj.list_mut(interpreter)?;
        lst.push(&mut interpreter.mem, result)?;
    }

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_index(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let value = unpacker.required(b"value")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        if item.equal_inner(interpreter, &value)? {
            return Ok(ShimValue::Integer(idx as i32));
        }
    }

    Ok(default.unwrap_or(ShimValue::None))
}

pub(crate) fn shim_list_insert(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.required(b"index")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let idx = index.integer()? as isize;

    let lst = obj.list_mut(interpreter)?;
    let len = lst.len();

    // Handle negative and out-of-bounds indices like Python
    let insert_idx = if idx < 0 {
        // Negative indices count from the end
        (len as isize + idx).max(0) as usize
    } else if idx as usize > len {
        // Positive indices beyond length append at the end
        len
    } else {
        idx as usize
    };

    // Add a new element at the end (this will resize if needed)
    lst.push(&mut interpreter.mem, ShimValue::None)?;

    // Shift elements to make room
    for i in (insert_idx..len).rev() {
        let val = lst.get(&interpreter.mem, i as isize)?;
        lst.set(&mut interpreter.mem, (i + 1) as isize, val)?;
    }

    // Insert the value
    lst.set(&mut interpreter.mem, insert_idx as isize, value)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_pop(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.optional(b"index");
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    let lst = obj.list_mut(interpreter)?;

    if lst.is_empty() {
        return Ok(default.unwrap_or(ShimValue::None));
    }

    // Determine which index to pop
    let pop_idx = if let Some(idx_val) = index {
        let idx = idx_val.integer()? as isize;
        lst.wrap_idx(idx)?
    } else {
        // Default to last element
        lst.len() - 1
    };

    // Get the value at the index
    let value = lst.get(&interpreter.mem, pop_idx as isize)?;

    // Shift elements after pop_idx to the left
    for i in pop_idx..(lst.len() - 1) {
        let next_val = lst.get(&interpreter.mem, (i + 1) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, next_val)?;
    }

    // Decrease the length
    lst.len = (lst.len() - 1).into();

    Ok(value)
}

pub(crate) fn shim_list_sorted(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a new list with the same elements
    let new_lst_val = interpreter.mem.alloc_list()?;
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item)?;
    }

    // Sort the new list using the existing sort logic
    let mut sort_args = ArgBundle::new();
    sort_args.args.push(new_lst_val);
    if let Some(k) = key {
        sort_args.kwargs.push((b"key".to_vec(), k));
    }
    shim_list_sort(interpreter, &sort_args)?;

    Ok(new_lst_val)
}

pub(crate) fn shim_list_reverse(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    let len = lst.len();
    for i in 0..(len / 2) {
        let left = lst.get(&interpreter.mem, i as isize)?;
        let right = lst.get(&interpreter.mem, (len - 1 - i) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, right)?;
        lst.set(&mut interpreter.mem, (len - 1 - i) as isize, left)?;
    }

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_reversed(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    // Create a new list with reversed elements
    let new_lst_val = interpreter.mem.alloc_list()?;
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in (0..lst.len()).rev() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item)?;
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_dict_keys(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter
        .mem
        .alloc_native(DictKeysIterator { dict: obj, idx: 0 })?)
}

pub(crate) fn shim_dict_values(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter
        .mem
        .alloc_native(DictValuesIterator { dict: obj, idx: 0 })?)
}

pub(crate) fn shim_dict_items(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter
        .mem
        .alloc_native(DictItemsIterator { dict: obj, idx: 0 })?)
}

pub(crate) fn shim_dict_pop(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    dict.pop(interpreter, key, default)
}

pub(crate) fn shim_dict_index_set(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    dict.set(interpreter, key, value)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_dict_index_get(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    dict.get(interpreter, key)
}

pub(crate) fn shim_dict_index_get_default(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let default = unpacker.optional(b"default").unwrap_or(ShimValue::None);
    unpacker.end()?;

    Ok(dict.get(interpreter, key).unwrap_or(default))
}

pub(crate) fn shim_dict_index_has(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    Ok(ShimValue::Bool(dict.get(interpreter, key).is_ok()))
}

pub(crate) fn shim_dict_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(dict.len() as i32))
}

pub(crate) fn shim_dict_shrink_to_fit(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    dict.shrink_to_fit(interpreter)?;
    Ok(ShimValue::None)
}

/// Default implementation of the `.format` method available on every
/// `ShimValue`. It is used by string interpolation (`"\(value)"`) and may be
/// overridden by structs or native types that define their own `format`
/// method. The default implementation renders the value's string
/// representation and does not accept any additional formatting arguments.
pub(crate) fn shim_format(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    let s = obj.to_string_mem(&interpreter.mem);
    interpreter.mem.alloc_str(s.as_bytes())
}

enum FloatAlign {
    Left,
    Center,
    Right,
}

enum FloatNotation {
    Lower,
    Upper,
}

/// Implementation of the `.format` method for floats. It supports a number of
/// formatting options (all optional, passed as keyword or positional
/// arguments after the value):
///
/// - `fill`: the single character used to pad empty space (defaults to `" "`)
/// - `align`: `"left"`, `"center"`, or `"right"` (defaults to `"right"`)
/// - `force_sign`: always show the `+`/`-` sign (defaults to `false`)
/// - `width`: the total width of the formatted string
/// - `precision`: the number of digits to show after the decimal point
/// - `notation`: `"e"` or `"E"` to force scientific notation
pub(crate) fn shim_float_format(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = match unpacker.required(b"obj")? {
        ShimValue::Float(f) => f,
        ShimValue::Integer(i) => i as f32,
        other => {
            return Err(format!(
                "float format expects a number, got {}",
                other.to_string_mem(&interpreter.mem)
            ));
        }
    };

    let fill = match unpacker.optional(b"fill") {
        Some(v) => {
            let bytes = v.string(interpreter)?;
            let s = std::str::from_utf8(bytes)
                .map_err(|_| "`fill` must be a valid utf-8 string".to_string())?;
            let mut chars = s.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => c,
                _ => return Err("`fill` must be a single character".to_string()),
            }
        }
        None => ' ',
    };

    let align = match unpacker.optional(b"align") {
        Some(v) => match v.string(interpreter)? {
            b"left" => FloatAlign::Left,
            b"center" => FloatAlign::Center,
            b"right" => FloatAlign::Right,
            other => {
                return Err(format!(
                    "`align` must be \"left\", \"center\", or \"right\", got {:?}",
                    debug_u8s(other)
                ));
            }
        },
        None => FloatAlign::Right,
    };

    let force_sign = match unpacker.optional(b"force_sign") {
        Some(ShimValue::Bool(b)) => b,
        Some(_) => return Err("`force_sign` must be a bool".to_string()),
        None => false,
    };

    let width = match unpacker.optional(b"width") {
        Some(ShimValue::Integer(i)) if i >= 0 => Some(i as usize),
        Some(_) => return Err("`width` must be a non-negative integer".to_string()),
        None => None,
    };

    let precision = match unpacker.optional(b"precision") {
        Some(ShimValue::Integer(i)) if i >= 0 => Some(i as usize),
        Some(_) => return Err("`precision` must be a non-negative integer".to_string()),
        None => None,
    };

    let notation = match unpacker.optional(b"notation") {
        Some(v) => match v.string(interpreter)? {
            b"e" => Some(FloatNotation::Lower),
            b"E" => Some(FloatNotation::Upper),
            other => {
                return Err(format!(
                    "`notation` must be \"e\" or \"E\", got {:?}",
                    debug_u8s(other)
                ));
            }
        },
        None => None,
    };

    unpacker.end()?;

    // Render the number itself (sign + digits) before applying any padding.
    let mut num = match (precision, &notation) {
        (Some(p), Some(FloatNotation::Lower)) => format!("{:.*e}", p, value),
        (Some(p), Some(FloatNotation::Upper)) => format!("{:.*E}", p, value),
        (Some(p), None) => format!("{:.*}", p, value),
        (None, Some(FloatNotation::Lower)) => format!("{:e}", value),
        (None, Some(FloatNotation::Upper)) => format!("{:E}", value),
        (None, None) => format_float(value),
    };

    if force_sign && !num.starts_with('-') {
        num.insert(0, '+');
    }

    // Pad to the requested width using the fill character and alignment.
    if let Some(width) = width {
        let len = num.chars().count();
        if len < width {
            let pad = width - len;
            let (left, right) = match align {
                FloatAlign::Left => (0, pad),
                FloatAlign::Right => (pad, 0),
                FloatAlign::Center => (pad / 2, pad - pad / 2),
            };
            let mut padded = String::with_capacity(num.len() + pad);
            for _ in 0..left {
                padded.push(fill);
            }
            padded.push_str(&num);
            for _ in 0..right {
                padded.push(fill);
            }
            num = padded;
        }
    }

    interpreter.mem.alloc_str(num.as_bytes())
}

pub(crate) fn shim_str_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(s.len() as i32))
}

pub(crate) fn shim_str_split(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    unpacker.end()?;

    let len = s.len();
    let out_val = interpreter.mem.alloc_list()?;
    let out = out_val.list_mut(interpreter)?;
    let mut idx: usize = 0;
    while idx < len {
        while idx < len && s[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= len {
            break;
        }
        let start = idx;
        while idx < len && !s[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let val = interpreter.mem.alloc_str(&s[start..idx])?;
        out.push(&mut interpreter.mem, val)?;
    }

    Ok(out_val)
}

pub(crate) fn shim_str_join(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let sep = binding.string(interpreter)?.to_vec();
    let iterable = unpacker.required(b"iterable")?;
    unpacker.end()?;

    let mut iter_args = ArgBundle::new();
    let iterator = match iterable.attr_call(b"iter", interpreter, &mut iter_args)? {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(&mut (pc as usize), iter_args, &mut new_env)?
        }
    };

    let mut out = Vec::new();
    let mut first = true;
    loop {
        let mut next_args = ArgBundle::new();
        let item = match iterator.attr_call(b"next", interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(&mut (pc as usize), next_args, &mut new_env)?
            }
        };
        if item.is_stop_iteration() {
            break;
        }
        if !first {
            out.extend_from_slice(&sep);
        }
        first = false;
        out.extend_from_slice(item.to_string_mem(&interpreter.mem).as_bytes());
    }

    interpreter.mem.alloc_str(&out)
}

pub(crate) fn shim_str_upper(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let out = s.to_ascii_uppercase();
    unpacker.end()?;

    if out == s {
        Ok(binding)
    } else {
        interpreter.mem.alloc_str(&out)
    }
}

pub(crate) fn shim_str_lower(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let out = s.to_ascii_lowercase();
    unpacker.end()?;

    if out == s {
        Ok(binding)
    } else {
        interpreter.mem.alloc_str(&out)
    }
}

pub(crate) fn shim_str_strip(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let trimmed = trim_bytes(s);
    let out = if trimmed.len() == s.len() {
        None
    } else {
        Some(trimmed.to_vec())
    };
    unpacker.end()?;

    match out {
        Some(bytes) => interpreter.mem.alloc_str(&bytes),
        None => Ok(binding),
    }
}

pub(crate) fn shim_str_remove_prefix(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let prefix_val = unpacker.required(b"prefix")?;
    let prefix = prefix_val.string(interpreter)?;
    let out = if !prefix.is_empty() && s.starts_with(prefix) {
        Some(s[prefix.len()..].to_vec())
    } else {
        None
    };
    unpacker.end()?;

    match out {
        Some(bytes) => interpreter.mem.alloc_str(&bytes),
        None => Ok(binding),
    }
}

pub(crate) fn shim_str_remove_suffix(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let suffix_val = unpacker.required(b"suffix")?;
    let suffix = suffix_val.string(interpreter)?;
    let out = if !suffix.is_empty() && s.ends_with(suffix) {
        Some(s[..s.len() - suffix.len()].to_vec())
    } else {
        None
    };
    unpacker.end()?;

    match out {
        Some(bytes) => interpreter.mem.alloc_str(&bytes),
        None => Ok(binding),
    }
}

pub(crate) fn shim_str_split_lines(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    unpacker.end()?;

    let out_val = interpreter.mem.alloc_list()?;
    let out = out_val.list_mut(interpreter)?;
    let mut start = 0usize;
    let mut idx = 0usize;
    while idx < s.len() {
        if s[idx] == b'\n' {
            let end = if idx > start && s[idx - 1] == b'\r' { idx - 1 } else { idx };
            let val = interpreter.mem.alloc_str(&s[start..end])?;
            out.push(&mut interpreter.mem, val)?;
            idx += 1;
            start = idx;
        } else if s[idx] == b'\r' {
            let val = interpreter.mem.alloc_str(&s[start..idx])?;
            out.push(&mut interpreter.mem, val)?;
            idx += 1;
            if idx < s.len() && s[idx] == b'\n' {
                idx += 1;
            }
            start = idx;
        } else {
            idx += 1;
        }
    }
    if start < s.len() {
        let val = interpreter.mem.alloc_str(&s[start..])?;
        out.push(&mut interpreter.mem, val)?;
    }

    Ok(out_val)
}

pub(crate) fn shim_str_contains(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    let needle_val = unpacker.required(b"needle")?;
    let needle = needle_val.string(interpreter)?.to_vec();
    unpacker.end()?;

    Ok(ShimValue::Bool(find_subslice(&s, &needle).is_some()))
}

pub(crate) fn shim_str_ends_with(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    let suffix_val = unpacker.required(b"suffix")?;
    let suffix = suffix_val.string(interpreter)?.to_vec();
    unpacker.end()?;

    Ok(ShimValue::Bool(s.ends_with(&suffix)))
}

pub(crate) fn shim_str_starts_with(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    let prefix_val = unpacker.required(b"prefix")?;
    let prefix = prefix_val.string(interpreter)?.to_vec();
    unpacker.end()?;

    Ok(ShimValue::Bool(s.starts_with(&prefix)))
}

pub(crate) fn shim_str_find(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    let needle_val = unpacker.required(b"needle")?;
    let needle = needle_val.string(interpreter)?.to_vec();
    unpacker.end()?;

    Ok(match find_subslice(&s, &needle) {
        Some(idx) => ShimValue::Integer(idx as i32),
        None => ShimValue::None,
    })
}

pub(crate) fn shim_str_lstrip(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let start = s.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(s.len());
    let out = if start == 0 { None } else { Some(s[start..].to_vec()) };
    unpacker.end()?;

    match out {
        Some(bytes) => interpreter.mem.alloc_str(&bytes),
        None => Ok(binding),
    }
}

pub(crate) fn shim_str_rstrip(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    let end = s.iter().rposition(|b| !b.is_ascii_whitespace()).map(|i| i + 1).unwrap_or(0);
    let out = if end == s.len() { None } else { Some(s[..end].to_vec()) };
    unpacker.end()?;

    match out {
        Some(bytes) => interpreter.mem.alloc_str(&bytes),
        None => Ok(binding),
    }
}

pub(crate) fn shim_str_replace(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?.to_vec();
    let old_val = unpacker.required(b"old")?;
    let old = old_val.string(interpreter)?.to_vec();
    let new_val = unpacker.required(b"new")?;
    let new = new_val.string(interpreter)?.to_vec();
    unpacker.end()?;

    if old.is_empty() {
        return Err("replace: old must not be empty".to_string());
    }
    if find_subslice(&s, &old).is_none() {
        return Ok(binding);
    }

    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < s.len() {
        if s[idx..].starts_with(&old) {
            out.extend_from_slice(&new);
            idx += old.len();
        } else {
            out.push(s[idx]);
            idx += 1;
        }
    }

    interpreter.mem.alloc_str(&out)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|window| window == needle)
}

pub(crate) fn get_type_name(value: &ShimValue) -> &'static str {
    match value {
        ShimValue::Uninitialized => "uninitialized",
        ShimValue::Unit => "unit",
        ShimValue::None => "none",
        ShimValue::StopIteration => "stop_iteration",
        ShimValue::Integer(_) => "int",
        ShimValue::Float(_) => "float",
        ShimValue::Bool(_) => "bool",
        ShimValue::Fn(_) => "function",
        ShimValue::BoundMethod(..) => "bound method",
        ShimValue::BoundNativeMethod(_) => "bound native method",
        ShimValue::NativeFn(_) => "native function",
        ShimValue::String(..) => "string",
        ShimValue::Tuple(..) => "tuple",
        ShimValue::List(_) => "list",
        ShimValue::Dict(_) => "dict",
        ShimValue::Set(_) => "set",
        ShimValue::StructDef(_) => "struct definition",
        ShimValue::Struct(..) => "struct",
        ShimValue::Native(_, _) => "native object",
        ShimValue::Environment(_) => "environment",
    }
}

fn trim_bytes(s: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = s.len();

    // Trim from start
    while start < end && s[start].is_ascii_whitespace() {
        start += 1;
    }

    // Trim from end
    while end > start && s[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &s[start..end]
}

fn parse_string_to<T: std::str::FromStr>(s: &[u8], type_name: &str) -> Result<T, String> {
    let trimmed = trim_bytes(s);
    unsafe {
        std::str::from_utf8_unchecked(trimmed)
            .parse::<T>()
            .map_err(|_| {
                let string_repr = std::str::from_utf8(s).unwrap_or("<invalid utf8>");
                format!("Cannot convert string '{}' to {}", string_repr, type_name)
            })
    }
}

pub(crate) fn shim_str(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let string_repr = value.to_string(interpreter);
    let bytes = string_repr.as_bytes();
    interpreter.mem.alloc_str(bytes)
}

/// Python-like `repr()`: like `str()`, but strings are rendered as quoted,
/// escaped literals (matching how values appear when nested inside a printed
/// container).
pub(crate) fn shim_repr(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let string_repr = value.to_repr_mem(&interpreter.mem);
    let bytes = string_repr.as_bytes();
    interpreter.mem.alloc_str(bytes)
}

pub(crate) fn shim_bool(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    Ok(ShimValue::Bool(value.is_truthy(interpreter)?))
}

pub(crate) fn shim_int(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Ok(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int").map(ShimValue::Integer)
        }
        _ => Err(format!("Cannot convert {} to int", get_type_name(&value))),
    }
}

pub(crate) fn shim_float(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Ok(ShimValue::Float(f)),
        ShimValue::Bool(b) => Ok(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float").map(ShimValue::Float)
        }
        _ => Err(format!("Cannot convert {} to float", get_type_name(&value))),
    }
}

pub(crate) fn shim_try_int(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Integer(i)),
        ShimValue::Float(f) => Some(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Some(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int")
                .map(ShimValue::Integer)
                .ok()
        }
        _ => None,
    };

    Ok(result.unwrap_or(ShimValue::None))
}

pub(crate) fn shim_try_float(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Some(ShimValue::Float(f)),
        ShimValue::Bool(b) => Some(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float")
                .map(ShimValue::Float)
                .ok()
        }
        _ => None,
    };

    Ok(result.unwrap_or(ShimValue::None))
}

pub(crate) fn shim_sqrt(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err(format!("sqrt: expected int or float, got {value:?}")),
    };
    Ok(ShimValue::Float(f.sqrt()))
}

pub(crate) fn shim_pow(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let base = unpacker.required(b"self")?;
    let exp = unpacker.required(b"exp")?;
    unpacker.end()?;
    match (base, exp) {
        (ShimValue::Integer(b), ShimValue::Integer(e)) => {
            if e >= 0 {
                // Saturate at i32::MIN/i32::MAX instead of panicking on overflow.
                Ok(ShimValue::Integer(b.saturating_pow(e as u32)))
            } else {
                Ok(ShimValue::Float((b as f32).powi(e)))
            }
        }
        (ShimValue::Integer(b), ShimValue::Float(e)) => Ok(ShimValue::Float((b as f32).powf(e))),
        (ShimValue::Float(b), ShimValue::Integer(e)) => Ok(ShimValue::Float(b.powi(e))),
        (ShimValue::Float(b), ShimValue::Float(e)) => Ok(ShimValue::Float(b.powf(e))),
        _ => Err("pow: expected int or float arguments".to_string()),
    }
}

pub(crate) fn shim_round(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f.round() as i32)),
        _ => Err("round: expected int or float".to_string()),
    }
}

pub(crate) fn shim_ceil(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f.ceil() as i32)),
        _ => Err("ceil: expected int or float".to_string()),
    }
}

pub(crate) fn shim_floor(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f.floor() as i32)),
        _ => Err("floor: expected int or float".to_string()),
    }
}

pub(crate) fn shim_signum(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i.signum())),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f.signum() as i32)),
        _ => Err("signum: expected int or float".to_string()),
    }
}

pub(crate) fn shim_recip(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("recip: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.recip()))
}

pub(crate) fn shim_frac(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(_) => return Ok(ShimValue::Float(0.0)),
        ShimValue::Float(f) => f,
        _ => return Err("frac: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.fract()))
}

pub(crate) fn shim_trunc(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Float(f.trunc())),
        _ => Err("trunc: expected int or float".to_string()),
    }
}

pub(crate) fn shim_sin(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("sin: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.sin()))
}

pub(crate) fn shim_cos(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("cos: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.cos()))
}

pub(crate) fn shim_tan(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("tan: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.tan()))
}

pub(crate) fn shim_asin(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("asin: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.asin()))
}

pub(crate) fn shim_acos(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("acos: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.acos()))
}

pub(crate) fn shim_atan(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("atan: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.atan()))
}

pub(crate) fn shim_atan2(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    let other = unpacker.required(b"other")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("atan2: expected int or float".to_string()),
    };
    let g = match other {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("atan2: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.atan2(g)))
}

pub(crate) fn shim_sinh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("sinh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.sinh()))
}

pub(crate) fn shim_cosh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("cosh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.cosh()))
}

pub(crate) fn shim_tanh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("tanh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.tanh()))
}

pub(crate) fn shim_asinh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("asinh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.asinh()))
}

pub(crate) fn shim_acosh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("acosh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.acosh()))
}

pub(crate) fn shim_atanh(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("atanh: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.atanh()))
}

pub(crate) fn shim_log2(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("log2: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.log2()))
}

pub(crate) fn shim_log10(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("log10: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.log10()))
}

pub(crate) fn shim_ln(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("ln: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.ln()))
}

pub(crate) fn shim_log(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    let base = unpacker.required(b"base")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("log: expected int or float".to_string()),
    };
    let b = match base {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("log: base must be int or float".to_string()),
    };
    Ok(ShimValue::Float(f.log(b)))
}

pub(crate) fn shim_to_degrees(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("to_degrees: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.to_degrees()))
}

pub(crate) fn shim_abs(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    Ok(
        match value {
            ShimValue::Integer(i) => ShimValue::Integer(i.saturating_abs()),
            ShimValue::Float(f) => ShimValue::Float(f.abs()),
            _ => return Err("abs: expected int or float".to_string()),
        }
    )
}

pub(crate) fn shim_to_radians(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    unpacker.end()?;
    let f = match value {
        ShimValue::Integer(i) => i as f32,
        ShimValue::Float(f) => f,
        _ => return Err("to_radians: expected int or float".to_string()),
    };
    Ok(ShimValue::Float(f.to_radians()))
}

/// Clamp function that's generic over any comparable type.
/// Prefers returning the original value when value == min or value == max
pub(crate) fn shim_clamp(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    let min = unpacker.required(b"min")?;
    let max = unpacker.required(b"max")?;
    unpacker.end()?;

    // If value is less than min, return min
    if let Ok(std::cmp::Ordering::Less) = compare_values(interpreter, &value, &min) { return Ok(min) }

    // If value is less than max, return max
    if let Ok(std::cmp::Ordering::Greater) = compare_values(interpreter, &value, &max) { return Ok(max) }

    // Otherwise it's between the range, so we can return the value itself
    Ok(value)
}

/// Returns true if clamping wouldn't change the value
pub(crate) fn shim_in_range(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    let min = unpacker.required(b"min")?;
    let max = unpacker.required(b"max")?;
    unpacker.end()?;

    // If value is less than min, return false
    if let Ok(std::cmp::Ordering::Less) = compare_values(interpreter, &value, &min) {
        return Ok(ShimValue::Bool(false))
    }

    // If value is less than max, return false
    if let Ok(std::cmp::Ordering::Greater) = compare_values(interpreter, &value, &max) {
        return Ok(ShimValue::Bool(false))
    }

    // Otherwise it's between the range, so we can return the value itself
    Ok(ShimValue::Bool(true))
}


pub(crate) fn shim_min(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    let min = unpacker.required(b"min")?;
    unpacker.end()?;
    match (value, min) {
        (ShimValue::Integer(v), ShimValue::Integer(m)) => Ok(ShimValue::Integer(v.min(m))),
        (ShimValue::Integer(v), ShimValue::Float(m)) => Ok(ShimValue::Float((v as f32).min(m))),
        (ShimValue::Float(v), ShimValue::Integer(m)) => Ok(ShimValue::Float(v.min(m as f32))),
        (ShimValue::Float(v), ShimValue::Float(m)) => Ok(ShimValue::Float(v.min(m))),
        _ => Err("pow: expected int or float arguments".to_string()),
    }
}


pub(crate) fn shim_max(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"self")?;
    let max = unpacker.required(b"max")?;
    unpacker.end()?;
    match (value, max) {
        (ShimValue::Integer(v), ShimValue::Integer(m)) => Ok(ShimValue::Integer(v.max(m))),
        (ShimValue::Integer(v), ShimValue::Float(m)) => Ok(ShimValue::Float((v as f32).max(m))),
        (ShimValue::Float(v), ShimValue::Integer(m)) => Ok(ShimValue::Float(v.max(m as f32))),
        (ShimValue::Float(v), ShimValue::Float(m)) => Ok(ShimValue::Float(v.max(m))),
        _ => Err("pow: expected int or float arguments".to_string()),
    }
}

/// A set is implemented as a thin wrapper over a dictionary whose keys are the
/// set's elements (the values are all `None`). The backing dict is allocated
/// lazily: an empty set has `dict_pos == 0` and no dict object behind it.
#[derive(Debug)]
pub struct ShimSet {
    pub(crate) dict_pos: u24
}

impl ShimSet {
    pub fn new() -> Self{
        Self {
            dict_pos: u24::from(0)
        }
    }

    /// Whether the backing dictionary has been allocated yet.
    fn has_dict(&self) -> bool {
        self.dict_pos != u24::from(0)
    }

    pub fn add(&mut self, interpreter: &mut Interpreter, item: ShimValue) -> Result<(), String> {
        if !self.has_dict() {
            self.dict_pos = interpreter.mem.alloc_dict_raw()?;
        }
        let binding = ShimValue::Dict(self.dict_pos);
        let d = binding.dict_mut(interpreter)?;
        d.set(interpreter, item, ShimValue::None)
    }

    pub fn remove(&mut self, interpreter: &mut Interpreter, item: ShimValue) -> Result<(), String> {
        if !self.has_dict() {
            Err(format!("Item {} not found in set", item.to_string_mem(&interpreter.mem)))
        } else {
            let binding = ShimValue::Dict(self.dict_pos);
            let d = binding.dict_mut(interpreter)?;
            // `pop` errors when the key is absent and no default is supplied.
            match d.pop(interpreter, item, None) {
                Ok(_) => Ok(()),
                Err(_) => Err(format!("Item {} not found in set", item.to_string_mem(&interpreter.mem))),
            }
        }
    }

    /// Like `remove`, but a no-op when the item is absent.
    pub fn discard(&mut self, interpreter: &mut Interpreter, item: ShimValue) -> Result<(), String> {
        if !self.has_dict() {
            return Ok(());
        }
        let binding = ShimValue::Dict(self.dict_pos);
        let d = binding.dict_mut(interpreter)?;
        let _ = d.pop(interpreter, item, Some(ShimValue::None));
        Ok(())
    }

    pub fn contains(&mut self, interpreter: &mut Interpreter, item: ShimValue) -> Result<bool, String> {
        if !self.has_dict() {
            Ok(false)
        } else {
            let binding = ShimValue::Dict(self.dict_pos);
            let d = binding.dict_mut(interpreter)?;
            match d.get(interpreter, item) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }

    pub fn len(&self, interpreter: &Interpreter) -> usize {
        if !self.has_dict() {
            0
        } else {
            let d: &ShimDict = unsafe { interpreter.mem.get(self.dict_pos) };
            d.len()
        }
    }

    pub fn clear(&mut self) {
        self.dict_pos = u24::from(0);
    }

    /// Collect the set's elements into an owned vector. Used by the set
    /// operations and equality, which need a snapshot independent of the
    /// backing dict.
    pub fn entries(&self, interpreter: &Interpreter) -> Vec<ShimValue> {
        if !self.has_dict() {
            return Vec::new();
        }
        let d: &ShimDict = unsafe { interpreter.mem.get(self.dict_pos) };
        d.entries_array(interpreter)
            .iter()
            .filter(|e| e.is_valid())
            .map(|e| e.key)
            .collect()
    }
}

/// Iterator over a set's elements, yielding each key of the backing dict.
pub(crate) struct SetIterator {
    pub(crate) set: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for SetIterator {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_set_iter_next(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("Can't provide positional args to SetIterator.next()".to_string());
                }

                let itr: &mut SetIterator = args.args[0].as_native(interpreter)?;
                let dict_pos = itr.set.set(interpreter)?.dict_pos;
                if dict_pos == u24::from(0) {
                    return Ok(ShimValue::StopIteration);
                }
                let dict: &ShimDict = unsafe { interpreter.mem.get(dict_pos) };
                let entries = dict.entries_array(interpreter);

                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].key;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }

                Ok(ShimValue::StopIteration)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_set_iter_next)?)
        } else if ident == b"iter" {
            fn shim_set_iter_iter(
                _interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter
                .mem
                .alloc_bound_native_fn(self_as_val, shim_set_iter_iter)?)
        } else {
            Err(format!(
                "Can't get_attr {} on {}",
                debug_u8s(ident),
                type_name::<Self>()
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.set]
    }
}

pub(crate) fn shim_set(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let iterable = unpacker.optional(b"iterable");
    unpacker.end()?;

    let result = interpreter.mem.alloc_set()?;

    if let Some(iterable) = iterable {
        let mut iter_args = ArgBundle::new();
        let iterator = match iterable
            .get_attr(interpreter, b"iter")?
            .call(interpreter, &mut iter_args)?
        {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(&mut (pc as usize), iter_args, &mut new_env)?
            }
        };

        let next_method = iterator.get_attr(interpreter, b"next")?;

        loop {
            let mut next_args = ArgBundle::new();
            let item = match next_method.call(interpreter, &mut next_args)? {
                CallResult::ReturnValue(val) => val,
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        next_args,
                        &mut new_env,
                    )?
                }
            };

            if item.is_stop_iteration() {
                break;
            }

            result.set_mut(interpreter)?.add(interpreter, item)?;
        }
    }

    Ok(result)
}

pub(crate) fn shim_set_add(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    obj.set_mut(interpreter)?.add(interpreter, item)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_set_remove(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    obj.set_mut(interpreter)?.remove(interpreter, item)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_set_discard(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    obj.set_mut(interpreter)?.discard(interpreter, item)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_set_has(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    Ok(ShimValue::Bool(obj.set_mut(interpreter)?.contains(interpreter, item)?))
}

pub(crate) fn shim_set_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(ShimValue::Integer(obj.set(interpreter)?.len(interpreter) as i32))
}

pub(crate) fn shim_set_clear(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    obj.set_mut(interpreter)?.clear();

    Ok(ShimValue::None)
}

pub(crate) fn shim_set_iter(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(SetIterator { set: obj, idx: 0 })?)
}

pub(crate) fn shim_set_union(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let other = unpacker.required(b"other")?;
    unpacker.end()?;

    let mut items = obj.set(interpreter)?.entries(interpreter);
    items.extend(other.set(interpreter)?.entries(interpreter));

    let result = interpreter.mem.alloc_set()?;
    for item in items {
        result.set_mut(interpreter)?.add(interpreter, item)?;
    }

    Ok(result)
}

pub(crate) fn shim_set_intersection(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let other = unpacker.required(b"other")?;
    unpacker.end()?;

    let items = obj.set(interpreter)?.entries(interpreter);

    let result = interpreter.mem.alloc_set()?;
    for item in items {
        if other.set_mut(interpreter)?.contains(interpreter, item)? {
            result.set_mut(interpreter)?.add(interpreter, item)?;
        }
    }

    Ok(result)
}

pub(crate) fn shim_set_difference(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let other = unpacker.required(b"other")?;
    unpacker.end()?;

    let items = obj.set(interpreter)?.entries(interpreter);

    let result = interpreter.mem.alloc_set()?;
    for item in items {
        if !other.set_mut(interpreter)?.contains(interpreter, item)? {
            result.set_mut(interpreter)?.add(interpreter, item)?;
        }
    }

    Ok(result)
}
