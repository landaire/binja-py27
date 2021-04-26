use std::collections::HashMap;
use std::fmt;
use std::{
    borrow::Cow,
    cell::RefCell,
    sync::{Arc, Mutex},
};

use architecture::Architecture;
use log::*;

use binaryninja::architecture::ArchitectureExt;
use binaryninja::architecture::CoreArchitecture;
use binaryninja::architecture::CustomArchitectureHandle;
use binaryninja::architecture::InstructionInfo;
use binaryninja::architecture::{
    FlagCondition, FlagRole, ImplicitRegisterExtend, Register as Reg, RegisterInfo,
};
use binaryninja::{
    architecture,
    custombinaryview::{self, BinaryViewTypeBase, CustomBinaryView, CustomBinaryViewType},
};
use binaryninja::{
    binaryview::{self, BinaryView, BinaryViewExt},
    custombinaryview::{BinaryViewType, CustomView},
    filemetadata::FileMetadata,
    platform::Platform,
    section::{Section, Semantics},
    segment::{Segment, SegmentBuilder},
    symbol::{Symbol, SymbolType},
};

use binaryninja::llil;
use binaryninja::llil::{
    Label, Liftable, LiftableWithSize, LiftedExpr, LiftedNonSSA, Mutable, NonSSA,
};

use pydis::{decode, opcode::Opcode, opcode::Python27};

const MAX_REG_NO: u32 = 33;
const REG_SP: u32 = MAX_REG_NO;
// python doesn't have a link register, however if we define one binja will
// render our stacks better, because it will use it for return addresses
// instead of using our stack
const REG_LR: u32 = MAX_REG_NO - 1;

const PYTHON27_MAGIC: [u8; 4] = [0x03, 0xF3, 0x0D, 0x0A];

type TargetOpcode = pydis::opcode::Python27;

#[derive(Copy, Clone)]
struct Register {
    id: u32,
}

impl Register {
    fn new(id: u32) -> Self {
        Self { id }
    }
}

impl Into<llil::Register<Register>> for Register {
    fn into(self) -> llil::Register<Register> {
        llil::Register::ArchReg(self)
    }
}

impl architecture::RegisterInfo for Register {
    type RegType = Self;

    fn parent(&self) -> Option<Self> {
        None
    }
    fn offset(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        8
    }

    fn implicit_extend(&self) -> ImplicitRegisterExtend {
        ImplicitRegisterExtend::NoExtend
    }
}

impl architecture::Register for Register {
    type InfoType = Self;

    fn name(&self) -> Cow<str> {
        match self.id {
            0..=31 => format!("reg{}", self.id).into(),
            REG_LR => "fake_lr".into(),
            REG_SP => "sp".into(),
            _ => panic!("bad register number"),
        }
    }

    fn info(&self) -> Self {
        *self
    }

    fn id(&self) -> u32 {
        self.id
    }
}

impl<'a> Liftable<'a, PythonArch> for Register {
    type Result = llil::ValueExpr;

    fn lift(
        il: &'a llil::Lifter<PythonArch>,
        reg: Self,
    ) -> llil::Expression<'a, PythonArch, Mutable, NonSSA<LiftedNonSSA>, Self::Result> {
        il.reg(reg.size(), reg)
    }
}

impl<'a> LiftableWithSize<'a, PythonArch> for Register {
    fn lift_with_size(
        il: &'a llil::Lifter<PythonArch>,
        reg: Self,
        size: usize,
    ) -> llil::Expression<'a, PythonArch, Mutable, NonSSA<LiftedNonSSA>, llil::ValueExpr> {
        #[cfg(debug_assertions)]
        {
            if reg.size() < size {
                warn!(
                    "il @ {:x} attempted to lift {} byte register as {} byte expr",
                    il.current_address(),
                    reg.size(),
                    size
                );
            }
        }

        il.reg(reg.size(), reg)
    }
}

impl fmt::Debug for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name().as_ref())
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct Flag;

impl architecture::Flag for Flag {
    type FlagClass = Self;

    fn name(&self) -> Cow<str> {
        unreachable!()
    }

    fn role(&self, _class: Option<Self::FlagClass>) -> FlagRole {
        unreachable!()
    }

    fn id(&self) -> u32 {
        unreachable!()
    }
}

impl architecture::FlagWrite for Flag {
    type FlagType = Self;
    type FlagClass = Self;

    fn name(&self) -> Cow<str> {
        unreachable!()
    }

    fn id(&self) -> u32 {
        unreachable!()
    }

    fn class(&self) -> Option<Self> {
        unreachable!()
    }

    fn flags_written(&self) -> Vec<Self::FlagType> {
        unreachable!()
    }
}

impl architecture::FlagClass for Flag {
    fn name(&self) -> Cow<str> {
        unreachable!()
    }

    fn id(&self) -> u32 {
        unreachable!()
    }
}

impl architecture::FlagGroup for Flag {
    type FlagType = Self;
    type FlagClass = Self;

    fn name(&self) -> Cow<str> {
        unreachable!()
    }

    fn id(&self) -> u32 {
        unreachable!()
    }

    fn flags_required(&self) -> Vec<Self::FlagType> {
        unreachable!()
    }

    fn flag_conditions(&self) -> HashMap<Self, FlagCondition> {
        unreachable!()
    }
}

struct PythonArch {
    handle: CoreArchitecture,
    custom_handle: CustomArchitectureHandle<PythonArch>,
}

impl architecture::Architecture for PythonArch {
    type Handle = CustomArchitectureHandle<Self>;

    type RegisterInfo = Register;
    type Register = Register;

    type Flag = Flag;
    type FlagWrite = Flag;
    type FlagClass = Flag;
    type FlagGroup = Flag;

    type InstructionTextContainer = Vec<architecture::InstructionTextToken>;

    fn endianness(&self) -> binaryninja::Endianness {
        binaryninja::Endianness::LittleEndian
    }

    fn address_size(&self) -> usize {
        8
    }

    fn default_integer_size(&self) -> usize {
        8
    }

    fn instruction_alignment(&self) -> usize {
        1
    }

    fn max_instr_len(&self) -> usize {
        10
    }

    fn opcode_display_len(&self) -> usize {
        self.max_instr_len()
    }

    fn associated_arch_by_addr(&self, _addr: &mut u64) -> CoreArchitecture {
        self.handle
    }

    fn instruction_info(&self, data: &[u8], addr: u64) -> Option<InstructionInfo> {
        use architecture::BranchInfo;

        debug!("decode {:#x}", addr);
        let mut cursor = std::io::Cursor::new(data);
        let instr = decode::<TargetOpcode, _>(&mut cursor).ok()?;
        let sz = instr.len();
        let pc = addr.wrapping_add(sz as u64);

        let mut res = InstructionInfo::new(sz, false);

        if !instr.opcode.is_jump() {
            return Some(res);
        }

        let raw_arg = instr.arg.unwrap();

        let target = if instr.opcode.is_absolute_jump() {
            instr.arg.unwrap() as u64
        } else {
            pc.wrapping_add(raw_arg as u64)
        };

        if instr.opcode.is_conditional_jump() {
            res.add_branch(BranchInfo::True(target), None);
            res.add_branch(BranchInfo::False(pc as u64), None);
        } else if instr.opcode.is_conditional_jump() {
            res.add_branch(BranchInfo::Unconditional(target), None);
        }

        if instr.opcode == TargetOpcode::CALL_FUNCTION {
            self.
            res.add_branch(BranchInfo::Call())
        }

        Some(res)
    }

    fn instruction_text(
        &self,
        data: &[u8],
        addr: u64,
    ) -> Option<(usize, Self::InstructionTextContainer)> {
        use architecture::InstructionTextToken;
        use architecture::InstructionTextTokenContents::*;

        debug!("decode {:#x}", addr);
        let mut cursor = std::io::Cursor::new(data);
        let instr = decode::<TargetOpcode, _>(&mut cursor).ok()?;
        let pc = addr.wrapping_add(instr.len() as u64);

        let mnem = format!("{:?}", instr.opcode);

        let mut res = Vec::new();

        res.push(InstructionTextToken::new(Instruction, mnem));

        // Get all the ops that have an argument, add a space
        if instr.arg.is_some() {
            res.push(InstructionTextToken::new(Text, " "));
        }

        if !instr.opcode.is_jump() {
            if let Some(arg) = instr.arg {
                // All arguments are integers
                res.push(InstructionTextToken::new(
                    Integer(arg as u64),
                    format!("{:#x}", arg),
                ));
            }
        } else {
            let raw_arg = instr.arg.unwrap();
            let target = if instr.opcode.is_absolute_jump() {
                instr.arg.unwrap() as u64
            } else {
                pc.wrapping_add(raw_arg as u64)
            };

            res.push(InstructionTextToken::new(
                CodeRelativeAddress(target),
                format!("{:#x}", target),
            ));
        }

        Some((instr.len(), res))
    }

    fn instruction_llil(
        &self,
        data: &[u8],
        addr: u64,
        il: &mut llil::Lifter<Self>,
    ) -> Option<(usize, bool)> {
        debug!("lifting {:#x}", addr);

        let mut cursor = std::io::Cursor::new(data);
        let instr = decode::<TargetOpcode, _>(&mut cursor).ok()?;
        let sz = instr.len();

        let pc = addr.wrapping_add(sz as u64);

        let mut cont = true;

        il.unimplemented().append();

        return Some((sz, cont));

        // match instr.opcode {
        //     Python27::BINARY_ADD => {
        //         il.push(8, il.add(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_AND => {
        //         il.push(8, il.and(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_DIVIDE => {
        //         il.push(8, il.divu(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_FLOOR_DIVIDE => {
        //         // TODO: How to represent this?
        //         il.push(8, il.divs(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_LSHIFT => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.lsl(
        //                 8,
        //                 il.reg(8, llil::Register::Temp(1)),
        //                 il.reg(8, llil::Register::Temp(0)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Python27::BINARY_MODULO => {
        //         il.push(8, il.modu(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_MULTIPLY => {
        //         il.push(8, il.mul(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_OR => {
        //         il.push(8, il.or(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_POWER => {
        //         il.unimplemented().append();
        //     }
        //     Python27::BINARY_RSHIFT => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.lsl(
        //                 8,
        //                 il.reg(8, llil::Register::Temp(1)),
        //                 il.reg(8, llil::Register::Temp(0)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Python27::BINARY_SUBSC => {
        //         il.set_reg(8, llil::Register::Temp(0), il.mul(8, il.pop(8), 8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.load(
        //                 8,
        //                 il.add(
        //                     8,
        //                     il.reg(8, llil::Register::Temp(1)),
        //                     il.reg(8, llil::Register::Temp(0)),
        //                 ),
        //             ),
        //         )
        //         .append();
        //     }
        //     Python27::BINARY_SUBTRACT => {
        //         il.push(8, il.sub(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_TRUE_DIVIDE => {
        //         il.push(8, il.divu(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BINARY_XOR => {
        //         il.push(8, il.xor(8, il.pop(8), il.pop(8))).append();
        //     }
        //     Python27::BREAK_LOOP => {
        //         il.unimplemented().append();
        //     }
        //     Python27::BUILD_CLASS => {

        //     }
        //     Python27::ROT_TWO => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(1))).append();
        //     }
        //     Python27::ROT_THREE => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(2), il.pop(8)).append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(1))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(2))).append();
        //     }
        //     Python27::POP_TOP => {
        //         il.pop(8);
        //     }
        // }

        // match op {
        //     Op::Addr(a) => il.push(8, il.load(8, a)).append(),
        //     Op::Deref => il.push(8, il.load(8, il.pop(8))).append(),
        //     Op::Const1u(v) => il.push(8, v as u64).append(),
        //     Op::Const1s(v) => il.push(8, v as u64).append(),
        //     Op::Const2u(v) => il.push(8, v as u64).append(),
        //     Op::Const2s(v) => il.push(8, v as u64).append(),
        //     Op::Const4u(v) => il.push(8, v as u64).append(),
        //     Op::Const4s(v) => il.push(8, v as u64).append(),
        //     Op::Const8u(v) | Op::Constu(v) => il.push(8, v as u64).append(),
        //     Op::Const8s(v) | Op::Consts(v) => il.push(8, v as u64).append(),
        //     Op::Dup => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //     }
        //     Op::Drop => il.pop(8).append(),
        //     Op::Over => {
        //         il.set_reg(
        //             8,
        //             llil::Register::Temp(0),
        //             il.load(8, il.add(8, il.reg(8, Register::new(REG_SP)), 8u64)),
        //         )
        //         .append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //     }
        //     Op::Pick(off) => {
        //         il.set_reg(
        //             8,
        //             llil::Register::Temp(0),
        //             il.load(
        //                 8,
        //                 il.add(
        //                     8,
        //                     il.reg(8, Register::new(MAX_REG_NO)),
        //                     il.mul(8, 8u64, off),
        //                 ),
        //             ),
        //         )
        //         .append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //     }
        //     Op::Swap => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(1))).append();
        //     }
        //     Op::Rot => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(2), il.pop(8)).append();

        //         il.push(8, il.reg(8, llil::Register::Temp(0))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(2))).append();
        //         il.push(8, il.reg(8, llil::Register::Temp(1))).append();
        //     }
        //     Op::Abs => {
        //         // bit hack for abs:
        //         // x = pop()
        //         // y = x >>> 63
        //         // push((x ^ y) - y)

        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(
        //             8,
        //             llil::Register::Temp(1),
        //             il.asr(8, il.reg(8, llil::Register::Temp(0)), 63),
        //         )
        //         .append();

        //         il.push(
        //             8,
        //             il.sub(
        //                 8,
        //                 il.xor(
        //                     8,
        //                     il.reg(8, llil::Register::Temp(0)),
        //                     il.reg(8, llil::Register::Temp(1)),
        //                 ),
        //                 il.reg(8, llil::Register::Temp(1)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Op::And => il.push(8, il.and(8, il.pop(8), il.pop(8))).append(),
        //     Op::Div => il.push(8, il.divu(8, il.pop(8), il.pop(8))).append(),
        //     Op::Minus => il.push(8, il.sub(8, il.pop(8), il.pop(8))).append(),
        //     Op::Mod => il.push(8, il.modu(8, il.pop(8), il.pop(8))).append(),
        //     Op::Mul => il.push(8, il.mul(8, il.pop(8), il.pop(8))).append(),
        //     Op::Neg => il.push(8, il.neg(8, il.pop(8))).append(),
        //     Op::Not => il.push(8, il.not(8, il.pop(8))).append(),
        //     Op::Or => il.push(8, il.or(8, il.pop(8), il.pop(8))).append(),
        //     Op::Plus => il.push(8, il.add(8, il.pop(8), il.pop(8))).append(),
        //     Op::PlusConst(v) => il.push(8, il.add(8, il.pop(8), v)).append(),
        //     Op::Shl => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.lsl(
        //                 8,
        //                 il.reg(8, llil::Register::Temp(1)),
        //                 il.reg(8, llil::Register::Temp(0)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Op::Shr => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.lsr(
        //                 8,
        //                 il.reg(8, llil::Register::Temp(1)),
        //                 il.reg(8, llil::Register::Temp(0)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Op::Shra => {
        //         il.set_reg(8, llil::Register::Temp(0), il.pop(8)).append();
        //         il.set_reg(8, llil::Register::Temp(1), il.pop(8)).append();

        //         il.push(
        //             8,
        //             il.asr(
        //                 8,
        //                 il.reg(8, llil::Register::Temp(1)),
        //                 il.reg(8, llil::Register::Temp(0)),
        //             ),
        //         )
        //         .append();
        //     }
        //     Op::Xor => il.push(8, il.xor(8, il.pop(8), il.pop(8))).append(),
        //     Op::Bra(off) => {
        //         let cond_expr = il.cmp_ne(8, il.pop(8), 0u64);

        //         let mut new_false: Option<Label> = None;
        //         let mut new_true: Option<Label> = None;

        //         let ft = pc;
        //         let tt = pc.wrapping_add(off as i64 as u64);

        //         {
        //             let f = il.label_for_address(ft).unwrap_or_else(|| {
        //                 new_false = Some(Label::new());
        //                 new_false.as_ref().unwrap()
        //             });

        //             let t = il.label_for_address(tt).unwrap_or_else(|| {
        //                 new_true = Some(Label::new());
        //                 new_true.as_ref().unwrap()
        //             });

        //             il.if_expr(cond_expr, t, f).append();
        //         }

        //         if let Some(t) = new_true.as_mut() {
        //             il.mark_label(t);

        //             il.jump(il.const_ptr(tt)).append();
        //         }

        //         if let Some(f) = new_false.as_mut() {
        //             il.mark_label(f);
        //         }
        //     }
        //     Op::Eq => il.push(8, il.cmp_e(8, il.pop(8), il.pop(8))).append(),
        //     Op::Ge => il.push(8, il.cmp_uge(8, il.pop(8), il.pop(8))).append(),
        //     Op::Gt => il.push(8, il.cmp_ugt(8, il.pop(8), il.pop(8))).append(),
        //     Op::Le => il.push(8, il.cmp_ule(8, il.pop(8), il.pop(8))).append(),
        //     Op::Lt => il.push(8, il.cmp_ult(8, il.pop(8), il.pop(8))).append(),
        //     Op::Ne => il.push(8, il.cmp_ne(8, il.pop(8), il.pop(8))).append(),
        //     Op::Skip(off) => {
        //         let target = pc.wrapping_add(off as i64 as u64);

        //         match il.label_for_address(target) {
        //             Some(l) => il.goto(l),
        //             None => il.jump(il.const_ptr(target)),
        //         }
        //         .append();

        //         cont = false;
        //     }
        //     Op::Lit(v) => il.push(8, v as u64).append(),
        //     Op::Reg(r) => il.push(8, il.reg(8, Register::new(r.into()))).append(),
        //     Op::BReg(_, _) => todo!(),
        //     Op::RegX(_) => todo!(),
        //     Op::BRegX(_, _) => todo!(),
        //     Op::DerefSize(sz) => il.push(8, il.zx(8, il.load(sz, il.pop(8)))).append(),
        //     Op::Nop => il.nop().append(),
        // };

        Some((sz, cont))
    }

    fn flag_write_llil<'a>(
        &self,
        _flag: Self::Flag,
        _flag_write: Self::FlagWrite,
        _op: llil::FlagWriteOp<Self::Register>,
        _il: &'a mut llil::Lifter<Self>,
    ) -> Option<LiftedExpr<'a, Self>> {
        None
    }

    fn flag_cond_llil<'a>(
        &self,
        _cond: FlagCondition,
        _class: Option<Self::Flag>,
        _il: &'a mut llil::Lifter<Self>,
    ) -> Option<LiftedExpr<'a, Self>> {
        None
    }

    fn flag_group_llil<'a>(
        &self,
        _group: Self::FlagGroup,
        _il: &'a mut llil::Lifter<Self>,
    ) -> Option<LiftedExpr<'a, Self>> {
        None
    }

    fn registers_all(&self) -> Vec<Self::Register> {
        (0..=MAX_REG_NO).map(|ii| Register::new(ii)).collect()
    }

    fn registers_full_width(&self) -> Vec<Self::Register> {
        self.registers_all()
    }

    fn registers_global(&self) -> Vec<Self::Register> {
        Vec::new()
    }

    fn registers_system(&self) -> Vec<Self::Register> {
        Vec::new()
    }

    fn flags(&self) -> Vec<Self::Flag> {
        Vec::new()
    }

    fn flag_write_types(&self) -> Vec<Self::FlagWrite> {
        Vec::new()
    }

    fn flag_classes(&self) -> Vec<Self::FlagClass> {
        Vec::new()
    }

    fn flag_groups(&self) -> Vec<Self::FlagGroup> {
        Vec::new()
    }

    fn flags_required_for_flag_condition(
        &self,
        _cond: FlagCondition,
        _class: Option<Flag>,
    ) -> Vec<Self::Flag> {
        Vec::new()
    }

    fn stack_pointer_reg(&self) -> Option<Self::Register> {
        Some(Register::new(REG_SP))
    }

    fn link_reg(&self) -> Option<Self::Register> {
        Some(Register::new(REG_LR))
    }

    fn register_from_id(&self, id: u32) -> Option<Self::Register> {
        match id {
            0..=MAX_REG_NO => Some(Register::new(id)),
            _ => None,
        }
    }

    fn flag_from_id(&self, _id: u32) -> Option<Self::Flag> {
        None
    }

    fn flag_write_from_id(&self, _id: u32) -> Option<Self::FlagWrite> {
        None
    }

    fn flag_class_from_id(&self, _id: u32) -> Option<Self::FlagClass> {
        None
    }

    fn flag_group_from_id(&self, _id: u32) -> Option<Self::FlagGroup> {
        None
    }

    fn handle(&self) -> CustomArchitectureHandle<Self> {
        self.custom_handle
    }
}

impl AsRef<CoreArchitecture> for PythonArch {
    fn as_ref(&self) -> &CoreArchitecture {
        &self.handle
    }
}

use binaryninja::callingconvention::*;

#[derive(Copy, Clone, Default, Eq, PartialEq, Hash)]
struct PythonCC {}

impl CallingConventionBase for PythonCC {
    type Arch = PythonArch;

    fn caller_saved_registers(&self) -> Vec<Register> {
        Vec::new()
    }

    fn callee_saved_registers(&self) -> Vec<Register> {
        Vec::new()
    }

    fn int_arg_registers(&self) -> Vec<Register> {
        Vec::new()
    }

    fn float_arg_registers(&self) -> Vec<Register> {
        Vec::new()
    }
    fn arg_registers_shared_index(&self) -> bool {
        false
    }

    fn reserved_stack_space_for_arg_registers(&self) -> bool {
        false
    }
    fn stack_adjusted_on_return(&self) -> bool {
        false
    }

    fn return_int_reg(&self) -> Option<Register> {
        None
    }
    fn return_hi_int_reg(&self) -> Option<Register> {
        None
    }
    fn return_float_reg(&self) -> Option<Register> {
        None
    }
    fn global_pointer_reg(&self) -> Option<Register> {
        None
    }

    fn implicitly_defined_registers(&self) -> Vec<Register> {
        Vec::new()
    }

    fn is_eligible_for_heuristics(&self) -> bool {
        true
    }
}

struct PythonViewType {
    binary_view_type: BinaryViewType,
}

impl AsRef<BinaryViewType> for PythonViewType {
    fn as_ref(&self) -> &BinaryViewType {
        &self.binary_view_type
    }
}

use binaryninja::binaryview::BinaryViewBase;
impl BinaryViewTypeBase for PythonViewType {
    fn is_valid_for(&self, data: &binaryview::BinaryView) -> bool {
        let mut magic = [0u8; 4];
        if data.read(&mut magic[..], 0) != 4 {
            false
        } else {
            magic == PYTHON27_MAGIC
        }
    }
}

impl CustomBinaryViewType for PythonViewType {
    fn create_custom_view<'builder>(
        &self,
        data: &binaryview::BinaryView,
        builder: custombinaryview::CustomViewBuilder<'builder, Self>,
    ) -> binaryview::Result<custombinaryview::CustomView<'builder>> {
        let mut python_data = vec![0u8; data.len() - 8];
        if data.read(&mut python_data, 8) != data.len() - 8 {
            error!("did not read enough data when constructing custom view");
            panic!("did not read enough data when constructing custom view");
        }
        println!("{:X?}", &python_data[..0xF]);

        if let py_marshal::Obj::Code(code) = py_marshal::read::marshal_loads(&python_data).unwrap()
        {
            builder.create::<PythonView>(data, code)
        } else {
            error!("bad python object");
            panic!("bad python object");
        }
    }
}

struct PythonView {
    binary_view: BinaryView,
    code: Option<Arc<py_marshal::Code>>,
}

impl AsRef<BinaryView> for PythonView {
    fn as_ref(&self) -> &BinaryView {
        &self.binary_view
    }
}

impl BinaryViewBase for PythonView {
    fn entry_point(&self) -> u64 {
        0
    }

    fn default_endianness(&self) -> binaryninja::Endianness {
        binaryninja::Endianness::BigEndian
    }

    fn address_size(&self) -> usize {
        8
    }
}

unsafe impl CustomBinaryView for PythonView {
    type Args = Arc<py_marshal::Code>;

    fn new(handle: binaryview::BinaryView, args: &Self::Args) -> binaryview::Result<Self> {
        Ok(PythonView {
            binary_view: handle,
            code: Some(Arc::clone(&args)),
        })
    }

    fn init(&self, args: Self::Args) -> binaryview::Result<()> {
        let parent_view = self.parent_view().unwrap();

        let arch = CoreArchitecture::by_name("Python27").expect("Python27 arch not registerred");
        let plat = arch.standalone_platform().unwrap();
        self.set_default_arch(&arch);
        self.set_default_platform(&plat);

        self.add_entry_point(&plat, 0);

        let main_segment = Segment::new(0x1e..0xf2 as u64)
            .parent_backing(0x1e..0xf2)
            .executable(true)
            .contains_code(true)
            .readable(true)
            .writable(true)
            .is_auto(true);

        self.add_segment(main_segment);

        let user_section = Section::new("Code", 0x1e..0xf2 as u64)
            .semantics(Semantics::ReadOnlyCode)
            .entry_size(self.code.as_ref().unwrap().code.len() as u64);

        self.add_section(user_section);

        let start = Symbol::new(SymbolType::Function, "_start", 0x1e).create();
        self.define_auto_symbol(&start);

        Ok(())
    }
}

fn register(name: &str, description: &str) {
    let arch = architecture::register_architecture(name, |custom_handle, core_arch| PythonArch {
        handle: core_arch,
        custom_handle: custom_handle,
    });

    let cc = register_calling_convention(arch, "default", PythonCC::default());
    arch.set_default_calling_convention(&cc);

    let python_view = custombinaryview::register_view_type(name, description, |binary_view_type| {
        PythonViewType { binary_view_type }
    });
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn CorePluginInit() -> bool {
    binaryninja::logger::init(log::LevelFilter::Trace).expect("Failed to set up logging");

    register("Python27", "Load Python 2.7 bytecode");

    true
}
