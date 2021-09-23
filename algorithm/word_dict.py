from collections import deque


class WordDict:
    """
    Given string(s) and dictionary of words dict,

    """

    """
    107
    determine if s can be broken into a sequence of one or more dictionary words.
    
    Input: s = "lintcode", dict = ["lint", "code"]
    Output: True
    (有多种可能使 word_break_dp(s, word_set) = True)
    Input："lintcode"，["de","ding","co","code","lint"]
    Output：True ( ["lint code", "lint co de"] )

    @param s: A strings
    @param wordSet: A dictionary of words dict
    @return: A boolean
    """
    def word_break_dp(self, s, word_set):
        if not s:
            return True
        if not word_set:
            return False

        n = len(s)
        dp = [False] * (n + 1)  # dp[i] 表示前 i 个字符是否能够被划分为若干个单词
        dp[0] = True

        max_length = max([len(word) for word in word_set]) if word_set else 0

        for i in range(1, n + 1):
            for j in range(1, max_length + 1):  # j 代表最后一个词的长度
                if i < j:
                    break
                if not dp[i - j]:
                    continue
                word = s[i - j:i]  # 长度为j
                if word in word_set:
                    dp[i] = True
                    break

        return dp[n]

    """
    582
    Given a string s and a dictionary of words dict, 
    add spaces in s to construct a sentence where each word is a valid dictionary word.
    Return all such possible sentences.
    
    Input: "helloworld", ["hell", "world", "o", "l", "he", "low", "orld", "hel"]
    Output: ["he l l o world","he l low orld","hel l o world","hel low orld","hell o world"]
    
    @param: s: string
    @param: word_dict: set of words.
    @return: list of strings - All possible sentences.
    """
    def word_break_sentences(self, s, word_dict):  # 记忆化搜索剪枝 memo + DFS
        memo = {}  # s -> [all possible partitions for s]
        return self.dfs_wbs(s, word_dict, memo)

    # 找到 s 的所有切割方案并 return
    def dfs_wbs(self, s, word_dict, memo):
        if s in memo:
            return memo[s]
        if len(s) == 0:
            return []

        partitions = []  # list of strings, each string is one sentence / partition

        # 取整个字符串s
        if s in word_dict:
            partitions.append(s)

        # 取字符串s的前缀子串作为word，word长度为1到len(s) - 1
        for i in range(1, len(s)):
            prefix = s[:i]
            if prefix not in word_dict:
                continue

            sub_partitions = self.dfs_wbs(s[i:], word_dict, memo)  # [all possible partitions for s[i:]]
            for sub_partition in sub_partitions:
                partitions.append(prefix + " " + sub_partition)

        memo[s] = partitions
        return partitions

    """
    683
    Given a string s and a dictionary of words dict, ignore cases
    add spaces in s to construct a sentence where each word is a valid dictionary word.
    Return the number of sentences you can form. 
    
    Input: "helloworld", ["hell", "world", "o", "l", "he", "low", "orld", "hel"]
    Output: 5 ( ["he l l o world","he l low orld","hel l o world","hel low orld","hell o world"] )

    @param: s: string
    @param: word_dict: set of words.
    @return: integer - the number of all possible sentences.
    """
    def word_break_nos(self, s, dict):  # number of sentences
        self.max_length = 0
        lower_dict = set()
        for word in dict:
            self.max_length = max(self.max_length, len(word))
            lower_dict.add(word.lower())

        return self.dfs_wbnos(s.lower(), 0, lower_dict, {})

    def dfs_wbnos(self, s, index, lower_dict, memo):
        if index == len(s):
            return 1

        if index in memo:
            return memo[index]

        memo[index] = 0
        for i in range(index, len(s)):
            if i + 1 - index > self.max_length:
                break
            word = s[index: i + 1]
            if word not in lower_dict:
                continue
            memo[index] += self.dfs_wbnos(s, i + 1, lower_dict, memo)

        return memo[index]

    """
    given a string word
    return list of all words that can get with one letter replacement
    """
    def get_next_words(self, word):
        next_words = []
        for i in range(len(word)):
            left, right = word[:i], word[i + 1:]
            # for char in 'abcdefghijklmnopqrstuvwxyz':
            # char above is the same as char below
            for j in range(26):
                char = chr(j + ord('a'))
                if word[i] == char:
                    continue
                next_words.append(left + char + right)
        return next_words

    """
    640
    @param s: a string
    @param t: a string
    @return: true if they are one edit distance apart or false
    """
    def is_one_edit_distance(self, s, t):
        ls, lt = len(s), len(t)
        if ls == lt:
            diff = 0
            for i in range(ls):
                if s[i] != t[i]:
                    diff += 1
            return diff == 1

        if ls == lt + 1:
            for i in range(lt):
                if s[i] != t[i]:
                    return s[i + 1:] == t[i:]
            return True

        if lt == ls + 1:
            for i in range(ls):
                if s[i] != t[i]:
                    return s[i:] == t[i + 1:]
            return True

        return False

    """
    120
    find the shortest transformation sequence from start to end, output the length of the sequence.
    Rules: 1. Only one letter can be changed at a time 2. Each intermediate word must exist in the dictionary. 
    (Start and end words do not need to appear in the dictionary)
    
    start = "hit"
    end = "cog"
    dict = ["hot","dot","dog","lot","log"]
    output = 5 ("hit"->"hot"->"dot"->"dog"->"cog")
    
    Return 0 if there is no such transformation sequence.
    All words have the same length.
    All words contain only lowercase alphabetic characters.
    You may assume no duplicates in the dictionary.
    
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def word_ladder_min_length(self, start, end, dict):
        # Bidirectional BFS
        if start == end:
            return 1
        q1, q2 = deque([start]), deque([end])
        steps = 1
        while q1 and q2:
            steps += 1
            len_q1 = len(q1)
            for _ in range(len_q1):
                word = q1.popleft()

                for new_word in self.get_next_words(word):
                    if new_word in q2:
                        return steps
                    if new_word in dict:
                        q1.append(new_word)
                        dict.remove(new_word)
            steps += 1
            len_q2 = len(q2)
            for _ in range(len_q2):
                word = q2.popleft()

                for new_word in self.get_next_words(word):
                    if new_word in q1:
                        return steps
                    if new_word in dict:
                        q2.append(new_word)
                        dict.remove(new_word)
        return 0

    """
    121
    given same as 120, 
    find all shortest transformation sequence(s) from start to end
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
    def find_ladders(self, start, end, dict):
        dict.add(start)
        dict.add(end)
        indexes = self.build_indexes(dict)

        distance = self.bfs(end, indexes)

        results = []
        self.dfs(start, end, distance, indexes, [start], results)

        return results

    def build_indexes(self, dict):
        indexes = {}
        for word in dict:
            for i in range(len(word)):
                key = word[:i] + '%' + word[i + 1:]
                if key in indexes:
                    indexes[key].add(word)
                else:
                    indexes[key] = set([word])
        return indexes

    def bfs(self, end, indexes):
        distance = {end: 0}
        queue = deque([end])
        while queue:
            word = queue.popleft()
            for next_word in self.get_next_words_with_indexes(word, indexes):
                if next_word not in distance:
                    distance[next_word] = distance[word] + 1
                    queue.append(next_word)
        return distance

    def get_next_words_with_indexes(self, word, indexes):
        words = []
        for i in range(len(word)):
            key = word[:i] + '%' + word[i + 1:]
            for w in indexes.get(key, []):
                words.append(w)
        return words

    def dfs(self, curt, target, distance, indexes, path, results):
        if curt == target:
            results.append(list(path))
            return

        for word in self.get_next_words_with_indexes(curt, indexes):
            if distance[word] != distance[curt] - 1:
                continue
            path.append(word)
            self.dfs(word, target, distance, indexes, path, results)
            path.pop()


if __name__ == '__main__':
    wd = WordDict()
    print(wd.word_break_dp("helloworld", ["hell", "world", "o"]))
    print(wd.word_break_sentences("zzz", ["z", "zz", "zzz"]))
    next_of_abc = wd.get_next_words("abc")
    print(next_of_abc)
    print(sum([not wd.is_one_edit_distance(n, "abc") for n in next_of_abc]))  # expect 0
    print(wd.word_ladder_min_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
    print(wd.find_ladders("hit", "cog", {"hot", "dot", "dog", "lot", "log"}))
