@njit(cache=False, nogil=True)
def corrected_eval(board, state, scratch, history, ply, corr):
    """The network's score, nudged by what searches from this pawn structure
    have found it to be worth. A structural misjudgement repeats in every
    line with the same pawns, so it can be learned during the game."""
    raw = evaluate_acc(board, state, scratch, history, ply)
    value = raw + corr[state[ST_TURN], state[ST_PKEY] & (CORR_SIZE - 1)] // CORR_GRAIN
    if value > MATE_THRESHOLD - 1:
        value = MATE_THRESHOLD - 1
    elif value < -MATE_THRESHOLD + 1:
        value = -MATE_THRESHOLD + 1
    return value


@njit(cache=False, nogil=True)
def update_correction(state, corr, depth, diff):
    index = state[ST_PKEY] & (CORR_SIZE - 1)
    turn = state[ST_TURN]
    weight = depth + 1 if depth < 15 else 16
    value = corr[turn, index]
    value = divide_toward_zero(value * (256 - weight) + diff * CORR_GRAIN * weight, 256)
    if value > CORR_LIMIT:
        value = CORR_LIMIT
    elif value < -CORR_LIMIT:
        value = -CORR_LIMIT
    corr[turn, index] = value


@njit(cache=False, nogil=True, inline="always")
def capture_slot_victim(board, move):
    if move_flags(move) & F_EP != 0:
        return PAWN
    return board[move_to(move)] & 7


@njit(cache=False, nogil=True)
def score_moves(board, moves, scores, count, tt_move, killers, history_heuristic,
                cont_history, cont_stack, capt_hist,
                turn, ply, counter_move):
    """Rate moves so the best are searched first: table move, winning
    captures (victim first, then what captures like it have done before),
    promotions, killers, counter move, quiet history, losing captures."""
    for i in range(count):
        move = moves[i]
        if move == tt_move:
            scores[i] = 1000000
            continue
        flags = move_flags(move)
        if flags & F_CAPTURE != 0:
            victim = capture_slot_victim(board, move)
            piece = board[move_from(move)]
            attacker = piece & 7
            rank = (VICTIM_SCORE[victim] * 16 - VICTIM_SCORE[attacker]
                    + capt_hist[piece, move_to(move), victim] // 8)
            losing = False
            if flags & (F_EP | F_PROMO) == 0 and SEE_VALUE[victim] < SEE_VALUE[attacker]:
                losing = see(board, move) < 0
            if losing:
                scores[i] = 200000 + rank
            else:
                scores[i] = 500000 + rank
        elif flags & F_PROMO != 0:
            scores[i] = 400000 + VICTIM_SCORE[move_promo(move)]
        elif move == killers[ply, 0]:
            scores[i] = 300000
        elif move == killers[ply, 1]:
            scores[i] = 290000
        elif move == counter_move and counter_move != 0:
            scores[i] = 280000
        else:
            value = history_heuristic[
                turn, square_index(move_from(move)) * 64 + square_index(move_to(move))
            ]
            slot = piece_slot(board[move_from(move)], move_to(move))
            if ply >= 1 and cont_stack[ply - 1] >= 0:
                value += cont_history[cont_stack[ply - 1], slot]
            if ply >= 2 and cont_stack[ply - 2] >= 0:
                value += cont_history[cont_stack[ply - 2], slot]
            if ply >= 4 and cont_stack[ply - 4] >= 0:
                value += cont_history[cont_stack[ply - 4], slot]
            scores[i] = value


@njit(cache=False, nogil=True, inline="always")
def pick_best(moves, scores, count, start):
    """Swap the best remaining move into position `start`."""
    best = start
    for i in range(start + 1, count):
        if scores[i] > scores[best]:
            best = i
    if best != start:
        moves[start], moves[best] = moves[best], moves[start]
        scores[start], scores[best] = scores[best], scores[start]


@njit(cache=False, nogil=True)
def is_repetition(hashes, hkey, ply, halfmove, game_keys, game_count):
    """Seen before -- either on the way here, or earlier in the real game?"""
    limit = ply - halfmove
    if limit < 0:
        limit = 0
    i = ply - 2
    while i >= limit:
        if hashes[i] == hkey[0]:
            return True
        i -= 2
    for j in range(game_count):
        if game_keys[j] == hkey[0]:
            return True
    return False


@njit(cache=False, nogil=True)
def quiescence(board, state, history, hashes, hkey, moves, scores, scratch,
               killers, history_heuristic,
               cont_history, cont_stack, counters, tt_key, tt_data, corr, capt_hist,
               alpha, beta, ply):
    """Captures only, until the position is quiet. In check, every evasion,
    and no standing pat: a side in check has not got the option of doing
    nothing."""
    counters[C_NODES] += 1
    if counters[C_NODES] > counters[C_LIMIT]:
        counters[C_ABORT] = 1
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate_acc(board, state, scratch, history, ply)

    # Any table entry answers a quiescence question, whatever its depth.
    if beta - alpha == 1:
        base = np.int64(hkey[0] & np.uint64(TT_BUCKETS - 1)) * TT_WAYS
        for way in range(TT_WAYS):
            if tt_key[base + way] == hkey[0]:
                slot = base + way
                score = tt_data[slot, 2]
                if score < MATE_THRESHOLD and score > -MATE_THRESHOLD:
                    flag = tt_data[slot, 1]
                    if (flag == EXACT or (flag == LOWER and score >= beta)
                            or (flag == UPPER and score <= alpha)):
                        return score
                break

    checked = in_check(board, state, state[ST_TURN])
    if checked:
        stand_pat = -MATE + ply
        best = stand_pat
    else:
        stand_pat = corrected_eval(board, state, scratch, history, ply, corr)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat

    count = generate(board, state, moves[ply], 0 if checked else 1)
    score_moves(board, moves[ply], scores[ply], count, 0, killers,
                history_heuristic, cont_history, cont_stack, capt_hist,
                state[ST_TURN], ply, 0)

    legal = 0
    for index in range(count):
        pick_best(moves[ply], scores[ply], count, index)
        move = moves[ply, index]
        if not checked:
            flags = move_flags(move)
            victim = capture_slot_victim(board, move)
            gain = VICTIM_SCORE[victim]
            if stand_pat + gain + 200 < alpha and (flags & F_PROMO) == 0:
                continue
            if flags & (F_EP | F_PROMO) == 0:
                attacker_kind = board[move_from(move)] & 7
                if SEE_VALUE[victim] < SEE_VALUE[attacker_kind] and see(board, move) < 0:
                    continue
        if not make_move(board, state, history, hashes, hkey, ply, move):
            continue
        legal += 1
        score = -quiescence(board, state, history, hashes, hkey, moves, scores, scratch,
                            killers, history_heuristic,
                            cont_history, cont_stack, counters, tt_key, tt_data,
                            corr, capt_hist, -beta, -alpha, ply + 1)
        unmake_move(board, state, history, hashes, hkey, ply)
        if counters[C_ABORT] != 0:
            return 0
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    if checked and legal == 0:
        return -MATE + ply
    return best


@njit(cache=False, nogil=True)
def history_update(board, moves, ply, index, move, depth, turn, history_heuristic,
                   cont_history, cont_stack, capt_hist, killers, counter_moves,
                   previous):
    """Credit the move that cut, and debit everything tried before it."""
    bonus = depth * depth
    if bonus > 1200:
        bonus = 1200
    flags = move_flags(move)
    if (flags & (F_CAPTURE | F_PROMO)) == 0:
        if killers[ply, 0] != move:
            killers[ply, 1] = killers[ply, 0]
            killers[ply, 0] = move
        if previous != 0:
            counter_moves[board[move_to(previous)], move_to(previous)] = move
        slot_h = square_index(move_from(move)) * 64 + square_index(move_to(move))
        history_heuristic[turn, slot_h] = gravity(history_heuristic[turn, slot_h], bonus)
        slot_c = piece_slot(board[move_from(move)], move_to(move))
        for back in (1, 2, 4):
            if ply < back or cont_stack[ply - back] < 0:
                continue
            cont_history[cont_stack[ply - back], slot_c] = gravity(
                cont_history[cont_stack[ply - back], slot_c], bonus)
    elif flags & F_CAPTURE != 0:
        victim = capture_slot_victim(board, move)
        piece = board[move_from(move)]
        capt_hist[piece, move_to(move), victim] = gravity(
            capt_hist[piece, move_to(move), victim], bonus)

    quiet_cut = (flags & (F_CAPTURE | F_PROMO)) == 0
    for j in range(index):
        earlier = moves[ply, j]
        eflags = move_flags(earlier)
        if (eflags & F_CAPTURE) != 0:
            victim = capture_slot_victim(board, earlier)
            piece = board[move_from(earlier)]
            capt_hist[piece, move_to(earlier), victim] = gravity(
                capt_hist[piece, move_to(earlier), victim], -bonus)
        elif quiet_cut and (eflags & F_PROMO) == 0:
            slot_h = (square_index(move_from(earlier)) * 64
                      + square_index(move_to(earlier)))
            history_heuristic[turn, slot_h] = gravity(history_heuristic[turn, slot_h], -bonus)
            slot_c = piece_slot(board[move_from(earlier)], move_to(earlier))
            for back in (1, 2, 4):
                if ply < back or cont_stack[ply - back] < 0:
                    continue
                cont_history[cont_stack[ply - back], slot_c] = gravity(
                    cont_history[cont_stack[ply - back], slot_c], -bonus)


@njit(cache=False, nogil=True)
def negamax(board, state, history, hashes, hkey, moves, scores, scratch,
            killers, history_heuristic,
            cont_history, cont_stack, tt_key, tt_data, counters, counter_moves,
            game_keys, game_count, evals, corr, capt_hist,
            depth, alpha, beta, ply, previous, excluded):
    """Alpha-beta in negamax form, principal variation search.

    `excluded` is the move a singular test is proving the rest of the list
    cannot match; while it is set the table is neither read for a cutoff nor
    written.
    """
    counters[C_NODES] += 1
    if counters[C_NODES] > counters[C_LIMIT]:
        counters[C_ABORT] = 1
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate_acc(board, state, scratch, history, ply)

    pv = beta - alpha > 1
    if ply > 0:
        if state[ST_HALF] >= 100:
            return 0
        if is_repetition(hashes, hkey, ply, state[ST_HALF], game_keys, game_count):
            return 0
        # Mate distance: nothing here can beat a mate already found nearer
        # the root.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    alpha_original = alpha
    base = np.int64(hkey[0] & np.uint64(TT_BUCKETS - 1)) * TT_WAYS
    slot = base
    found = False
    for way in range(TT_WAYS):
        if tt_key[base + way] == hkey[0]:
            slot = base + way
            found = True
            break
    tt_move = 0
    tt_depth = -1
    tt_score = 0
    tt_flag = -1
    if found:
        tt_move = tt_data[slot, 3]
        tt_depth = tt_data[slot, 0]
        tt_score = tt_data[slot, 2]
        tt_flag = tt_data[slot, 1]
        if tt_score > MATE_THRESHOLD:
            tt_score -= ply
        elif tt_score < -MATE_THRESHOLD:
            tt_score += ply
        if tt_depth >= depth and ply > 0 and excluded == 0:
            if tt_flag == EXACT:
                return tt_score
            if tt_flag == LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == UPPER and tt_score <= alpha:
                return tt_score

    checked = in_check(board, state, state[ST_TURN])
    if checked and ply < MAX_PLY - 8:
        depth += 1

    # Internal iterative reduction: no table move means ordering starts from
    # nothing, so search a ply shallower and leave a move for next time.
    if tt_move == 0 and depth >= 4 and excluded == 0:
        depth -= 1
    if depth <= 0:
        return quiescence(board, state, history, hashes, hkey, moves, scores, scratch,
                          killers, history_heuristic,
                          cont_history, cont_stack, counters, tt_key, tt_data,
                          corr, capt_hist, alpha, beta, ply)

    # Static evaluation at every node out of check: the pruning below reads
    # it, and so does the next-but-one ply to know whether we are improving.
    improving = False
    static = NO_EVAL
    eval_used = 0
    if checked:
        evals[ply] = NO_EVAL
    else:
        if excluded != 0:
            static = evals[ply]
        else:
            static = corrected_eval(board, state, scratch, history, ply, corr)
            evals[ply] = static
        eval_used = static
        # A table bound on the right side of the static score is a better
        # estimate of it.
        if found and tt_score < MATE_THRESHOLD and tt_score > -MATE_THRESHOLD:
            if ((tt_flag == LOWER and tt_score > static)
                    or (tt_flag == UPPER and tt_score < static) or tt_flag == EXACT):
                eval_used = tt_score
        if ply >= 2 and evals[ply - 2] != NO_EVAL:
            improving = static > evals[ply - 2]
        elif ply >= 4 and evals[ply - 4] != NO_EVAL:
            improving = static > evals[ply - 4]

    futile = False
    if (not checked and not pv and excluded == 0 and ply > 0
            and beta < MATE_THRESHOLD and beta > -MATE_THRESHOLD):
        # Reverse futility: far enough ahead that giving up material still
        # beats beta.
        margin = 85 * depth - (55 if improving else 0)
        if depth <= 7 and eval_used - margin >= beta:
            return (eval_used + beta) // 2
        if depth <= 4 and eval_used + 110 * depth + 90 <= alpha:
            futile = True
        # Razoring: so far behind that even generous margins do not reach
        # alpha. Ask quiescence directly, and trust it if it agrees.
        if depth <= 3 and eval_used + 150 * depth + 100 < alpha:
            razor = quiescence(board, state, history, hashes, hkey, moves, scores,
                               scratch, killers, history_heuristic,
                               cont_history, cont_stack, counters, tt_key, tt_data,
                               corr, capt_hist, alpha - 1, alpha, ply)
            if counters[C_ABORT] != 0:
                return 0
            if razor < alpha:
                return razor

        # Null move, only when already standing above beta: a pass that is
        # still good enough means the real moves almost certainly are.
        if (depth >= 3 and eval_used >= beta and previous != 0
                and has_non_pawn(board, state[ST_TURN])):
            reduction = 3 + depth // 4
            gap = (eval_used - beta) // 200
            reduction += gap if gap < 2 else 2
            make_null(state, history, hashes, hkey, ply)
            cont_stack[ply] = -1
            null_depth = depth - 1 - reduction
            if null_depth <= 0:
                null_score = -quiescence(board, state, history, hashes, hkey, moves,
                                         scores, scratch, killers, history_heuristic,
                                         cont_history, cont_stack, counters, tt_key,
                                         tt_data, corr, capt_hist,
                                         -beta, -beta + 1, ply + 1)
            else:
                null_score = -negamax(board, state, history, hashes, hkey, moves, scores,
                                      scratch, killers, history_heuristic,
                                      cont_history, cont_stack, tt_key, tt_data, counters,
                                      counter_moves, game_keys, game_count, evals, corr,
                                      capt_hist, null_depth, -beta, -beta + 1,
                                      ply + 1, 0, 0)
            unmake_null(state, history, hashes, hkey, ply)
            if counters[C_ABORT] != 0:
                return 0
            if null_score >= beta:
                return beta if null_score >= MATE_THRESHOLD else null_score

    count = generate(board, state, moves[ply], 0)
    counter = 0
    if previous != 0:
        counter = counter_moves[board[move_to(previous)], move_to(previous)]
    score_moves(board, moves[ply], scores[ply], count, tt_move, killers,
                history_heuristic, cont_history, cont_stack, capt_hist,
                state[ST_TURN], ply, counter)

    best_score = -INFINITY
    best_move = 0
    searched = 0
    quiet_seen = 0
    turn = state[ST_TURN]
    lmp_limit = 3 + depth * depth
    if not improving:
        lmp_limit = lmp_limit // 2 + 1

    for index in range(count):
        pick_best(moves[ply], scores[ply], count, index)
        move = moves[ply, index]
        if move == excluded:
            continue
        flags = move_flags(move)
        quiet = (flags & (F_CAPTURE | F_PROMO)) == 0
        order_score = scores[ply, index]
        if quiet:
            quiet_seen += 1

        # Pruning ahead of the search: never before one move has been
        # searched, never in check, never while every score is a mate.
        if ply > 0 and searched > 0 and not checked and best_score > -MATE_THRESHOLD:
            if quiet:
                if depth <= 7 and quiet_seen > lmp_limit:
                    continue
                if futile:
                    continue
                if (depth <= 6 and (flags & F_CASTLE) == 0
                        and see(board, move) < -45 * depth):
                    continue
            elif (depth <= 6 and order_score < 400000
                  and (flags & (F_EP | F_PROMO)) == 0
                  and see(board, move) < -90 * depth):
                continue

        extension = 0
        if (move == tt_move and excluded == 0 and ply > 0
                and depth >= SINGULAR_MIN_DEPTH
                and tt_depth >= depth - SINGULAR_TT_MARGIN
                and tt_flag != UPPER
                and tt_score < MATE_THRESHOLD and tt_score > -MATE_THRESHOLD):
            singular_beta = tt_score - SINGULAR_MARGIN * depth
            for j in range(count):
                moves[SHADOW + ply, j] = moves[ply, j]
                scores[SHADOW + ply, j] = scores[ply, j]
            verdict = negamax(board, state, history, hashes, hkey, moves, scores,
                              scratch, killers, history_heuristic,
                              cont_history, cont_stack, tt_key, tt_data,
                              counters, counter_moves, game_keys, game_count,
                              evals, corr, capt_hist,
                              (depth - 1) // 2, singular_beta - 1, singular_beta,
                              ply, previous, move)
            for j in range(count):
                moves[ply, j] = moves[SHADOW + ply, j]
                scores[ply, j] = scores[SHADOW + ply, j]
            if not checked:
                evals[ply] = static
            if counters[C_ABORT] != 0:
                return 0
            if verdict < singular_beta:
                extension = 1
            elif singular_beta >= beta:
                # Multi-cut: even without the table move something else
                # beats beta, so this node fails high whatever we pick.
                return singular_beta
            elif tt_score >= beta:
                # Not singular, and the table already expects a cutoff:
                # the table move does not deserve full depth here.
                extension = -1

        cont_stack[ply] = piece_slot(board[move_from(move)], move_to(move))
        if not make_move(board, state, history, hashes, hkey, ply, move):
            continue

        new_depth = depth - 1 + extension
        if searched == 0:
            score = -negamax(board, state, history, hashes, hkey, moves, scores, scratch,
                             killers, history_heuristic,
                             cont_history, cont_stack, tt_key, tt_data, counters,
                             counter_moves, game_keys, game_count, evals, corr, capt_hist,
                             new_depth, -beta, -alpha, ply + 1, move, 0)
        else:
            reduction = 0
            if (depth >= 3 and not checked
                    and searched >= (3 if pv else 2)
                    and (quiet or order_score < 400000)):
                row = depth if depth < 63 else 63
                column = searched if searched < 63 else 63
                reduction = LMR_TABLE[row, column]
                if not pv:
                    reduction += 1
                if not improving:
                    reduction += 1
                if quiet:
                    if order_score >= 280000:
                        reduction -= 1
                    else:
                        adjust = order_score // 12000
                        if adjust > 2:
                            adjust = 2
                        elif adjust < -2:
                            adjust = -2
                        reduction -= adjust
                if in_check(board, state, state[ST_TURN]):
                    reduction -= 1
                if reduction > new_depth - 1:
                    reduction = new_depth - 1
                if reduction < 0:
                    reduction = 0
            score = -negamax(board, state, history, hashes, hkey, moves, scores, scratch,
                             killers, history_heuristic,
                             cont_history, cont_stack, tt_key, tt_data, counters,
                             counter_moves, game_keys, game_count, evals, corr, capt_hist,
                             new_depth - reduction, -alpha - 1, -alpha, ply + 1, move, 0)
            # A reduced search that beats alpha has to prove it at full
            # depth before it is believed, even in a null window.
            if score > alpha and reduction > 0:
                score = -negamax(board, state, history, hashes, hkey, moves, scores,
                                 scratch, killers, history_heuristic,
                                 cont_history, cont_stack, tt_key, tt_data, counters,
                                 counter_moves, game_keys, game_count, evals, corr,
                                 capt_hist, new_depth, -alpha - 1, -alpha,
                                 ply + 1, move, 0)
            if score > alpha and score < beta:
                score = -negamax(board, state, history, hashes, hkey, moves, scores,
                                 scratch, killers, history_heuristic,
                                 cont_history, cont_stack, tt_key, tt_data, counters,
                                 counter_moves, game_keys, game_count, evals, corr,
                                 capt_hist, new_depth, -beta, -alpha,
                                 ply + 1, move, 0)
        unmake_move(board, state, history, hashes, hkey, ply)
        if counters[C_ABORT] != 0:
            return 0
        searched += 1

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            history_update(board, moves, ply, index, move, depth, turn,
                           history_heuristic, cont_history, cont_stack, capt_hist,
                           killers, counter_moves, previous)
            break

    if searched == 0:
        if excluded != 0:
            return alpha
        return -MATE + ply if checked else 0

    if best_score <= alpha_original:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    else:
        flag = EXACT

    # Teach the correction table what the search found here, when the
    # result says something about the static score: not in check, not a
    # tactical best move, and a bound that points the informative way.
    if (not checked and excluded == 0 and static != NO_EVAL
            and best_score < MATE_THRESHOLD and best_score > -MATE_THRESHOLD
            and (best_move == 0 or (move_flags(best_move) & (F_CAPTURE | F_PROMO)) == 0)
            and not (flag == LOWER and best_score <= static)
            and not (flag == UPPER and best_score >= static)):
        update_correction(state, corr, depth, best_score - static)

    stored = best_score
    if stored > MATE_THRESHOLD:
        stored += ply
    elif stored < -MATE_THRESHOLD:
        stored -= ply
    generation = counters[C_GEN]
    victim = base
    if found:
        victim = slot
    else:
        best_rank = 1 << 30
        for way in range(TT_WAYS):
            here = base + way
            if tt_key[here] == np.uint64(0):
                victim = here
                break
            rank = tt_data[here, 0]
            if tt_data[here, 4] != generation:
                rank -= 1 << 16
            if rank < best_rank:
                best_rank = rank
                victim = here
    slot = victim
    if excluded == 0 and (tt_key[slot] != hkey[0]
            or depth >= tt_data[slot, 0]
            or tt_data[slot, 4] != generation):
        tt_key[slot] = hkey[0]
        tt_data[slot, 0] = depth
        tt_data[slot, 1] = flag
        tt_data[slot, 2] = stored
        tt_data[slot, 3] = best_move
        tt_data[slot, 4] = generation
    return best_score


@njit(cache=False, nogil=True)
def search_root(board, state, history, hashes, hkey, moves, scores, scratch,
                killers, history_heuristic,
                cont_history, cont_stack, tt_key, tt_data, counters, counter_moves,
                game_keys, game_count, evals, corr, capt_hist,
                depth, alpha, beta, out):
    """One pass at `depth` inside the window [alpha, beta]. `out[1]` is
    fail-soft, so the caller can see which side it fell out of."""
    acc_refresh(board, history, 0)
    if in_check(board, state, state[ST_TURN]):
        evals[0] = NO_EVAL
    else:
        evals[0] = corrected_eval(board, state, scratch, history, 0, corr)
    count = generate(board, state, moves[0], 0)
    score_moves(board, moves[0], scores[0], count, out[0], killers,
                history_heuristic, cont_history, cont_stack, capt_hist,
                state[ST_TURN], 0, 0)
    best_move = 0
    best_score = -INFINITY
    searched = 0
    for index in range(count):
        pick_best(moves[0], scores[0], count, index)
        move = moves[0, index]
        cont_stack[0] = piece_slot(board[move_from(move)], move_to(move))
        if not make_move(board, state, history, hashes, hkey, 0, move):
            continue
        if searched == 0:
            score = -negamax(board, state, history, hashes, hkey, moves, scores, scratch,
                             killers, history_heuristic,
                             cont_history, cont_stack, tt_key, tt_data, counters,
                             counter_moves, game_keys, game_count, evals, corr, capt_hist,
                             depth - 1, -beta, -alpha, 1, move, 0)
        else:
            score = -negamax(board, state, history, hashes, hkey, moves, scores, scratch,
                             killers, history_heuristic,
                             cont_history, cont_stack, tt_key, tt_data, counters,
                             counter_moves, game_keys, game_count, evals, corr, capt_hist,
                             depth - 1, -alpha - 1, -alpha, 1, move, 0)
            if score > alpha and score < beta:
                score = -negamax(board, state, history, hashes, hkey, moves, scores,
                                 scratch, killers, history_heuristic,
                                 cont_history, cont_stack, tt_key, tt_data, counters,
                                 counter_moves, game_keys, game_count, evals, corr,
                                 capt_hist, depth - 1, -beta, -alpha, 1, move, 0)
        unmake_move(board, state, history, hashes, hkey, 0)
        if counters[C_ABORT] != 0:
            return 0
        searched += 1
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    out[0] = best_move
    out[1] = best_score
    return searched
