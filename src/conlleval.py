from collections import defaultdict, namedtuple
from typing import List, Tuple
import logging

LOGGER = logging.getLogger("baseline_hsupcon")

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


Metrics = namedtuple('Metrics', 'tp fp fn prec rec f1')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0      # TP
        self.correct_tags = 0       # Đúng token
        self.pred_chunks = 0        # TP + FP
        self.gold_chunks = 0        # TP + FN
        self.token_counter = 0      # Tổng số token

        # Đếm chi tiết theo từng loại thực thể
        self.tps_by_type = defaultdict(int)
        self.fps_by_type = defaultdict(int)
        self.fns_by_type = defaultdict(int)
        self.pred_chunks_by_type = defaultdict(int)
        self.gold_chunks_by_type = defaultdict(int)


def parse_tag(t: str) -> Tuple[str, str]:
    if t == 'O':
        return ('O', '')
    parts = t.split('-', 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return (t, '')


def split_tag(chunk_tag: str) -> Tuple[str, str]:
    """Phân tách tiền tố B/I/E/S và loại thực thể."""
    if chunk_tag == 'O':
        return ('O', '')
    if '-' in chunk_tag:
        prefix, chunk_type = chunk_tag.split('-', 1)
        return (prefix, chunk_type)
    return ('', chunk_tag)


def is_chunk_end(prev_tag: str, tag: str) -> bool:
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 in ('E', 'S'):
        return True
    if prefix1 in ('B', 'I') and (prefix2 in ('B', 'S', 'O') or chunk_type1 != chunk_type2):
        return True
    return False


def is_chunk_start(prev_tag: str, tag: str) -> bool:
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 in ('B', 'S'):
        return True
    if prefix2 in ('I', 'E') and (prefix1 in ('O', 'E', 'S') or chunk_type1 != chunk_type2):
        return True
    return False


def calc_metrics(tp: int, p: int, g: int) -> Tuple[float, float, float]:
    """Tính Precision, Recall, F1 từ số lượng đếm cơ bản."""
    prec = 0.0 if p == 0 else tp / p
    rec = 0.0 if g == 0 else tp / g
    f1 = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec) / (prec + rec)
    return prec, rec, f1


def evaluate(true_seqs: List[str], pred_seqs: List[str], verbose: bool = True) -> Tuple[float, float, float]:
    """
    Bộ đánh giá CoNLL-2003 Strict Entity-level F1 chuẩn xác 100%.
    Đếm đơn 1x cho cả Gold Phrases, Pred Phrases và Correct Phrases.
    """
    counts = EvalCounts()
    in_correct = False
    last_correct_type = ''
    last_correct_end = 0

    last_true_type = ''
    last_pred_type = ''

    for i, (true_tag, pred_tag) in enumerate(zip(true_seqs, pred_seqs)):
        counts.token_counter += 1
        if true_tag == pred_tag:
            counts.correct_tags += 1

        curr_true_end = is_chunk_end(last_true_type, true_tag)
        curr_pred_end = is_chunk_end(last_pred_type, pred_tag)
        curr_true_start = is_chunk_start(last_true_type, true_tag)
        curr_pred_start = is_chunk_start(last_pred_type, pred_tag)

        if in_correct:
            if curr_true_end and curr_pred_end and (split_tag(last_true_type)[1] == split_tag(last_pred_type)[1]):
                in_correct = False
                counts.correct_chunk += 1
                counts.tps_by_type[split_tag(last_correct_type)[1]] += 1
            elif curr_true_end != curr_pred_end or true_tag != pred_tag:
                in_correct = False

        if curr_true_start and curr_pred_start and (split_tag(true_tag)[1] == split_tag(pred_tag)[1]):
            in_correct = True
            last_correct_type = true_tag
            last_correct_end = i

        if curr_true_start:
            counts.gold_chunks += 1
            counts.gold_chunks_by_type[split_tag(true_tag)[1]] += 1

        if curr_pred_start:
            counts.pred_chunks += 1
            counts.pred_chunks_by_type[split_tag(pred_tag)[1]] += 1

        last_true_type = true_tag
        last_pred_type = pred_tag

    # Kiểm tra thực thể cuối cùng của câu
    if in_correct and is_chunk_end(last_true_type, 'O') and is_chunk_end(last_pred_type, 'O'):
        counts.correct_chunk += 1
        counts.tps_by_type[split_tag(last_correct_type)[1]] += 1

    # Tính Precision, Recall, F1 toàn cục (Strict Micro-F1)
    prec, rec, f1 = calc_metrics(counts.correct_chunk, counts.pred_chunks, counts.gold_chunks)
    acc = 0.0 if counts.token_counter == 0 else counts.correct_tags / counts.token_counter

    if verbose:
        LOGGER.info(
            f"Processed {counts.token_counter} tokens with {counts.gold_chunks} gold phrases; "
            f"Found: {counts.pred_chunks} phrases; Correct: {counts.correct_chunk}."
        )
        LOGGER.info(
            f"Token Accuracy: {acc:.4f} | Strict Precision: {prec:.4f} | Strict Recall: {rec:.4f} | Strict F1: {f1:.4f}"
        )
        # In chi tiết từng loại thực thể
        all_types = sorted(list(set(list(counts.gold_chunks_by_type.keys()) + list(counts.pred_chunks_by_type.keys()))))
        for ent_type in all_types:
            if not ent_type:
                continue
            tp_t = counts.tps_by_type[ent_type]
            p_t = counts.pred_chunks_by_type[ent_type]
            g_t = counts.gold_chunks_by_type[ent_type]
            pr_t, rc_t, f1_t = calc_metrics(tp_t, p_t, g_t)
            LOGGER.info(
                f"{ent_type:>18}: Precision: {pr_t:.4f}; Recall: {rc_t:.4f}; F1-score: {f1_t:.4f} "
                f"(Gold: {g_t}, Pred: {p_t})"
            )

    return prec, rec, f1

