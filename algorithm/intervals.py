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
    920
    check if a list of intervals have overlap
    @param intervals: an array of meeting time intervals
    @return: boolean - if a person could attend all meetings
    """
    def no_overlap(self, intervals):
        if not intervals:
            return True

        sl = sorted(intervals, key=lambda x: x.start)
        for i in range(len(sl) - 1):
            if sl[i + 1].start < sl[i].end:
                return False

        return True

    """
    156
    merge all overlapping intervals
    Input:  [(1,3),(2,6),(8,10),(15,18)]
    Output: [(1,6),(8,10),(15,18)]
    """
    def merge_intervals(self, intervals):
        if not intervals:
            return []

        sl = sorted(intervals, key=lambda x: x.start)

        last, output = None, []
        for interval in sl:
            if not last or last.end < interval.start:  # no overlap
                output.append(interval)
                last = interval
            else:
                last.end = max(last.end, interval.end)  # only update the last.end

        return output

    def merge_intervals_2moving_window(self, intervals):
        if len(intervals) <= 1:
            return intervals
        sl = sorted(intervals, key=lambda x: x.start)
        i = 1
        while i < len(sl):
            if sl[i - 1].end >= sl[i].start:
                new_start = min(sl[i - 1].start, sl[i].start)
                new_end = max(sl[i - 1].end, sl[i].end)
                sl = sl[:i - 1] + [Interval(new_start, new_end)] + sl[i + 1:]  # slice O(k)
            else:
                i += 1
        return sl

    def merge_intervals_dfs(self, intervals):
        if len(intervals) <= 1:
            return intervals
        sl = sorted(intervals, key=lambda x: x.start)
        return self.dfs(sl)

    def dfs(self, sorted_list):
        if len(sorted_list) <= 1:
            return sorted_list
        if sorted_list[1].start <= sorted_list[0].end:
            new_start = min(sorted_list[0].start, sorted_list[1].start)
            new_end = max(sorted_list[0].end, sorted_list[1].end)
            return self.dfs([Interval(new_start, new_end)] + sorted_list[2:])
        else:
            return [sorted_list[0]] + self.dfs(sorted_list[1:])

    """
    919
    Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
    find the minimum number of conference rooms required.
    @param intervals: an array of meeting time intervals
    @return: int - minimum number of conference rooms required
    """
    def thickest_overlaps(self, intervals):
        if not intervals:
            return 0

        sl = sorted(intervals, key = lambda x: x.start)
        # return max([sum([i in range(interval.start, interval.end) for interval in intervals])
        #               for i in range(sl[0].start, sl[-1].end)])

        rm = 0
        rmend = []

        for i in sl:
            if rm == 0:
                rmend.append(i.end)
                rm += 1
            elif i.start < min(rmend):
                rmend.append(i.end)
                rm += 1
            else:
                rmend[rmend.index(min(rmend))] = i.end

        return rm

    def thickest_overlaps_dp(self, intervals):
        if not intervals:
            return 0

        time = [0] * 500001
        for interval in intervals:
            time[interval.start] += 1
            time[interval.end] -= 1

        x = 0
        ans = 0
        for i in time:
            x += i
            ans = max(ans, x)

        return ans

    """
    1889
    given two string intervals (in lexicographic order), please judge whether the two intervals can be merged.
    If two intervals A and B satisfy that A ⋃ B is a continuous interval, then A and B can be merged.
    
    input："(b,c)" "[a,b]"
    output：true
    input："[a,b)" "(b,c]"
    output：false
    input: "[a,b]" "[ba,c)"
    output: true
    
    @param interval_A: a string represent a interval.
    @param interval_B: a string represent a interval.
    @return: if two intervals can merge return true, otherwise false.
    """
    def string_intervals_can_merge(self, interval_A, interval_B):
        first_ch_A = interval_A.split(interval_A[0])[1].split(',')[0]
        last_ch_A = interval_A.split(',')[1].split(interval_A[-1])[0]
        first_ch_B = interval_B.split(interval_B[0])[1].split(',')[0]
        last_ch_B = interval_B.split(',')[1].split(interval_B[-1])[0]
        if first_ch_A > first_ch_B:  # 排序，interval_B在左，所以比较last_ch_B和first_ch_A
            if last_ch_B > first_ch_A:
                return True
            if (last_ch_B == first_ch_A) and ((interval_B[-1] == ']') or (interval_A[0] == '[')):  # 不同时为开区间就行
                return True
            if ((last_ch_B + "a") == first_ch_A) and (interval_B[-1] == ']') and (interval_A[0] == '['):  # 同时为闭区间
                return True
            return False
        else:  # interval_A在左，所以比较last_ch_A和first_ch_B
            if last_ch_A > first_ch_B:
                return True
            if (last_ch_A == first_ch_B) and ((interval_A[-1] == ']') or (interval_B[0] == '[')):
                return True
            if ((last_ch_A + "a") == first_ch_B) and (interval_A[-1] == ']') and (interval_B[0] == '['):
                return True
            return False


if __name__ == '__main__':
    itvs = Intervals()
    print(itvs.no_overlap([Interval(465,497), Interval(386,462), Interval(354,380), Interval(134,189), Interval(199,282),
                           Interval(18,104), Interval(499,562), Interval(4,14), Interval(111,129), Interval(292,345)]))
    print(itvs.thickest_overlaps([Interval(65,424), Interval(351,507), Interval(314,807), Interval(387,722),
                                  Interval(19,797), Interval(259,722), Interval(165,221), Interval(136,897)]))  # 7
    print(itvs.thickest_overlaps_dp([Interval(65, 424), Interval(351, 507), Interval(314, 807), Interval(387, 722),
                                     Interval(19, 797), Interval(259, 722), Interval(165, 221), Interval(136, 897)]))

    merged_dfs = itvs.merge_intervals_dfs([Interval(1,3), Interval(2,6), Interval(8,10), Interval(15,18)])
    for i in merged_dfs:
        print((i.start, i.end), end=', ')

    print()

    merged = itvs.merge_intervals_2moving_window([Interval(1, 3), Interval(2, 6), Interval(8, 10), Interval(15, 18)])
    for i in merged:
        print((i.start, i.end), end=', ')

    print()

    merged = itvs.merge_intervals([Interval(1, 3), Interval(2, 6), Interval(8, 10), Interval(15, 18)])
    for i in merged:
        print((i.start, i.end), end=', ')
