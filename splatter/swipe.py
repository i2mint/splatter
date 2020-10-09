from heapq import heappushpop, heappush


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


class KeepMaxK(list):
    def __init__(self, k):
        super(self.__class__, self).__init__()
        self.k = k

    def push(self, item):
        if len(self) >= self.k:
            heappushpop(self, item)
        else:
            heappush(self, item)


class KeepMaxUnikK(object):
    def __init__(self, k):
        self.min_val_items = KeepMaxK(k)
        self.item_set = set()

    def push(self, item, val):
        if item not in self.item_set:
            self.item_set.add(item)
            self.min_val_items.push((val, item))

    def items_sorted(self):
        dists, items = list(zip(*self.min_val_items))
        # TODO: perhaps next two lines in a single pass?
        argsort_dists = sorted(range(len(dists)), key=dists.__getitem__)
        return [items[i] for i in argsort_dists]
        # Before was:
        # from numpy import array, argsort
        # return array(items)[argsort(dists)]


class KeepMinK(list):
    """
    Does what KeepMaxK does, but with min.
    NOTE: Only works with items that are pairs. This is because handling the more general case makes the push two
    times slower (overhead due to handling various cases).
    If you try to push items that are not list-like, it will raise a TypeError.
    If you push items that have only one element, it will raise an IndexError.
    If you push items that have more than 2 elements, only the first two will be taken into account.
    """

    def __init__(self, k):
        super(self.__class__, self).__init__()
        self.k = k

    def push(self, item):
        # try:
        #     item = [-item[0]] + list(item[1:])
        # except TypeError:
        #     item = -item

        if len(self) >= self.k:
            heappushpop(self, (-item[0], item[1]))
        else:
            heappush(self, (-item[0], item[1]))

    def get_list(self):
        return [(-item[0], item[1]) for item in self]


class HighestScoreSwipe(object):
    def __init__(self, score_of, chk_size, chk_step=1):
        self.score_of = score_of
        self.chk_size = chk_size
        self.chk_step = chk_step

    def __call__(self, it):
        pass


def highest_score_swipe(it, score_of=None, k=1, info_of=None, output=None):
    if score_of is None:
        score_of = lambda x: x

    km = KeepMaxK(k=k)

    if info_of is None:
        for x in it:
            km.push((score_of(x), x))
    else:
        if info_of == 'idx':
            for i, x in enumerate(it):
                km.push((score_of(x), i))
        else:
            assert callable(info_of), "info_of needs to be a callable (if not None or 'idx')"
            for x in it:
                km.push((score_of(x), info_of(x)))

    if output is None:
        return km
    elif isinstance(output, str):
        if output == 'top_tuples':
            return sorted(km, reverse=True)
        elif output == 'items':
            return [x[1] for x in km]
        elif output == 'scores':
            return [x[0] for x in km]
        elif output == 'top_score_items':
            return [x[1] for x in sorted(km, key=lambda x: x[0])]
        else:
            raise ValueError("Unrecognized output: ".format(output))
    else:
        return km
