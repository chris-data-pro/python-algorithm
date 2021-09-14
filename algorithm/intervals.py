# Write a function about interval


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Intervals:
    """
    Interval implementations
    meeting rooms problems
    """

    """
    30
    Given a non-overlapping interval list which is sorted by start point.
    Insert a new interval into it, 
    Return a list still in order and non-overlapping, merge intervals if necessary
    """
    def insert(self, sorted_intervals, new_interval):
        if not sorted_intervals:
            return

        start = new_interval.start
        end = new_interval.end
        left, right = [], []
        for interval in sorted_intervals:
            if start > interval.end:
                left.append(interval)
            elif end < interval.start:
                right.append(interval)
            else:
                start = min(start, interval.start)
                end = max(end, interval.end)

        return left + [Interval(start, end)] + right

    """
    
    """
