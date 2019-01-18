
def every_subseq(seq, cnt):
    """ Modifies cnt in place """
    L = len(seq)
    for i in xrange(L - 1):
        for j in xrange(i + 1, L):
            s = tuple(seq[i:j + 1])
            count = cnt.get(s, 0)
            cnt[s] = count + 1


def find_all_subseqs_naive(seqs, min_len=1):
    cnt = {}
    for seq in seqs:
        every_subseq(seq, cnt)

    to_del = []
    for el in cnt:
        if cnt[el] < min_len:
            to_del.append(el)
    for el in to_del:
        del cnt[el]
    return list(cnt.iteritems())


def validate_sub(s, open_instr_str, close_instr_str):
    counter = 0
    for c in s:
        if c in open_instr_str:
            counter += 1
        if c in close_instr_str:
            counter -= 1
    return counter == 0
