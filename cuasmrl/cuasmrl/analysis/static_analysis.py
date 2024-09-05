import os
from cuasmrl.utils.logger import get_logger

logger = get_logger(__name__)


def static_analysis(
    kernel_section,
    engine,
    ban_ops,
    memory_ops,
    min_st_analysis,  # out
    black_list,  # out
    st_db,
):
    # pre-scan to obtain assembly file stats
    debug = False
    if os.getenv("SIP_DEBUG", "0") == "1":
        debug = True

    # determine which lines are possible to mutate
    # e.g. LDG, STG, and they should not cross the boundary of a label or
    # LDGDEPBAR or BAR.SYNC or rw dependencies
    # lines = []
    candidates = []
    kernel_lineno_cnt = 0
    mem_loc = {}
    max_src_len = 0

    # analysis-only
    num_mem_inst = 0
    num_infer_only = 0
    num_in_db = 0
    # analysis-only

    for i, line in enumerate(kernel_section):
        line = line.strip()
        # skip headers
        if len(line) > 0 and line[0] == '[':
            ctrl_code, _, predicate, opcode, dst, src, _ = engine.decode(line)
            if ctrl_code is None:
                # a label
                continue

            kernel_lineno_cnt += 1

            # integeralize memory location
            if dst not in mem_loc:
                mem_loc[dst] = len(mem_loc)
            for s in src:
                if s not in mem_loc:
                    mem_loc[s] = len(mem_loc)
            max_src_len = max(max_src_len, len(src))

            # determine if MemOp;
            # opcode is like: LDG.E.128.SYS; i.e. {inst}.{modifier*}
            ban = False
            for op in ban_ops:
                # if op in opcode:
                if opcode.startswith(op):
                    ban = True
                    break
            if ban:
                if debug:
                    logger.warning(f'ban {ctrl_code} {opcode}')
                continue

            is_mem = False
            for op in memory_ops:
                # if op in opcode:
                if opcode.startswith(op):
                    if debug:
                        logger.info(f'mutable {ctrl_code} {opcode}')
                    candidates.append(i)
                    # lines.append(line)
                    is_mem = True
                    break
            if is_mem:
                num_mem_inst += 1
                resolved, tmp_opcode = find_def_use(
                    kernel_section,
                    engine,
                    min_st_analysis,
                    i,
                    line,
                    src,
                    debug,
                )
            if is_mem and resolved:
                if tmp_opcode is not None and tmp_opcode in st_db:
                    num_in_db += 1
                else:
                    num_infer_only += 1
            # XXX a hack for blacklist
            if is_mem and not resolved:
                black_list.add(line)
                candidates.pop(-1)

    print()
    logger.info('stall count analysis: ')
    # remove = []
    # for k, v in min_st_analysis.items():
    #     if v > 20:
    #         remove.append(k)
    #         logger.warning(f'pruning {k} -> {v}')
    #     elif k.startswith('LDS'):
    #         remove.append(k)
    #         logger.warning(f'pruning {k} -> {v}')
    #     else:
    #         logger.info(f'{k} -> {v}')
    # for k in remove:
    #     min_st_analysis.pop(k)
    for k, v in min_st_analysis.items():
        logger.info(f'{k} -> {v}')
    logger.info(
        f'num_black_list={len(black_list)}; {num_infer_only=}; {num_in_db=}; {num_mem_inst=}'
    )
    print()

    # dimension of the optimization problem
    dims = len(candidates)
    return candidates, dims, kernel_lineno_cnt, mem_loc, max_src_len


def find_def_use(
    kernel_section,
    engine,
    min_st_analysis,  # out
    idx,
    line,
    src,
    debug,
):

    resolved = False
    resolved_opcode = None

    for src_loc in src:
        if src_loc.startswith('UR'):
            # XXX can always skip uniform register?
            continue

        if src_loc.startswith('P'):
            # TODO what about predicate register
            pass

        j = 1
        accum = 0
        # print('line: ', line)
        while True:
            tmp_ctrl, *_, tmp_opcode, tmp_dst, tmp_src, _ = engine.decode(
                kernel_section[idx - j].strip())
            if tmp_ctrl is None:
                # if it is a label, don't care stall count
                logger.warning(f'reach a label before resolving users; {line}')

                # FIXME should break? just skip?
                break

            *_, stall_count = engine.decode_ctrl_code(tmp_ctrl)
            stall_count = int(stall_count[1:-1])
            accum += stall_count

            # print(self.kernel_section[idx - j].strip())
            # print(tmp_dst)

            if src_loc == tmp_dst:
                if tmp_opcode in min_st_analysis:
                    # logger.info(f'updating {line} with {accum} and {min_st_analysis[tmp_opcode]}')
                    min_st_analysis[tmp_opcode] = min(
                        min_st_analysis[tmp_opcode], accum)
                else:
                    # logger.info(f'adding {line} with {accum}')
                    min_st_analysis[tmp_opcode] = accum
                logger.info(f'resolve {tmp_opcode}')
                resolved = True
                resolved_opcode = tmp_opcode
                break

            j += 1
            if j >= 50:
                # logger.warning(
                #     f'cannot resolve stall count {line} for {src_loc}')
                # all_resolved = False
                break
                # raise RuntimeError(f'cannot reolve stall count {line}')
        if resolved:
            break

    if not resolved:
        logger.warning(f'cannot resolve stall count {line} for {src}')

    return resolved, resolved_opcode
