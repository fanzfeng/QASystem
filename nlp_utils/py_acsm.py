# -*- coding: utf-8 -*-
# @Author  : guoqi
'''
desc: match string and mode string

fail_transitions: 最长后缀字串的重匹配，节约匹配计算

'''

from collections import deque


class ACTreeNodePickleable:
    def __init__(self):
        self.go_to = {}
        self.output = []
        self.fail_transition_state = None


class ACTree:
    """
    example:
    acsm = ACTreePickleable(['cash', 'shew', 'ew'])
    text = 'cashew cashew'
    matches = acsm.search(text)
    print(matches)
    """
    def __init__(self, patterns, print_detail=False):
        self.print_detail = print_detail
        self.nodes = []
        self.state_max = 0
        self.nodes.append(ACTreeNodePickleable())
        self.add_patterns(patterns)
        self.set_fail_transitions()

    def add_patterns(self, patterns):
        for pattern in patterns:
            self.add_pattern(pattern)

    def add_pattern(self, pattern):
        current_state = 0
        j = 0
        next_state = self.nodes[current_state].go_to.get(pattern[j], None)
        while next_state is not None:
            current_state = next_state
            j += 1
            if j < len(pattern):
                next_state = self.nodes[current_state].go_to.get(pattern[j], None)
            else:
                break
        for i in range(j, len(pattern)):
            self.nodes.append(ACTreeNodePickleable())
            self.state_max += 1
            self.nodes[current_state].go_to.update({pattern[i]: self.state_max})
            current_state = self.state_max
        self.nodes[current_state].output.append(pattern)

    def set_fail_transitions(self):
        if self.print_detail:
            print("Begin set fail tran...")
        q = deque()
        for ch, state in self.nodes[0].go_to.items():
            q.append(state)
            self.nodes[state].fail_transition_state = 0
        while q:
            r = q.popleft()
            # goto calcu fail_tran
            for ch, state in self.nodes[r].go_to.items():
                q.append(state)
                k = 1
                fail_state = self.nodes[r].fail_transition_state# 上个结点的失败回转结点
                if self.print_detail:
                    print(" Cur state {} Bak state {} & str {} fail tran {}th to state{}".format(state, r, ch, k,
                            fail_state), " ", self.nodes[fail_state].go_to)
                while self.nodes[fail_state].go_to.get(ch, None) is None and fail_state != 0:
                    fail_state = self.nodes[fail_state].fail_transition_state
                    k += 1
                    if self.print_detail:
                        print(" Cur state {} Bak state {} & str {} fail tran {}th to state{}".format(state, r, ch, k,
                                fail_state), self.nodes[fail_state].go_to)
                # ch-->state, 下一个结点的错误回转结点为历史结点中同字符的目标结点
                # 保证最近匹配上的一个字符与目标结点的源字符是一致的（源字符index目标结点）
                self.nodes[state].fail_transition_state = self.nodes[fail_state].go_to.get(ch, 0)
                # 匹配成功的时候不用回转，直接输出所有匹配结果
                if self.nodes[self.nodes[state].fail_transition_state].output:
                    self.nodes[state].output += self.nodes[self.nodes[state].fail_transition_state].output

    def search(self, text):
        # 匹配至索引j，匹配串为s[0:j+1]，如果匹配失败则回退成s[1:j+1]，若再失败则s[2:j+1]，这个机制对应着模式串构成的树中的回退机制，沿着
        # 回退索引路径自动完成
        if self.print_detail:
            print("Search result: ")
        current_state = 0
        patterns_found = []
        # return match_ix, pattern
        for i in range(len(text)):
            while self.nodes[current_state].go_to.get(text[i], None) is None and current_state != 0:
                current_state = self.nodes[current_state].fail_transition_state
            current_state = self.nodes[current_state].go_to.get(text[i], 0)
            for pattern in self.nodes[current_state].output:
                patterns_found.append((i - len(pattern) + 1, pattern))
            if self.print_detail:
                print(text[i], "  ", self.nodes[current_state].output)
        return patterns_found

    def easy_search(self, text_input):
        """
        test fanzeng
        """
        patterns_found = []
        for j in range(0,len(text_input)-1):
            current_state = 0
            text = text_input[j:]
            for i in range(len(text)):
                print("w: {}   goto: {}".format(text[i], current_state))
                if text[i] not in self.nodes[current_state].go_to:
                    break
                else:
                    current_state = self.nodes[current_state].go_to.get(text[i])
                if current_state > 0 and len(self.nodes[current_state].output) > 0:
                    for pattern in self.nodes[current_state].output:
                        patterns_found.append((i - len(pattern) + 1 + j, pattern))
        return patterns_found


if __name__ == "__main__":
    modes = ["a", "ab", "bab", "bc", "bca", "c", "caa"]
    ac = ACTree(modes)
    print("Print Tree: ")
    for i in range(len(ac.nodes)):
        c = ac.nodes[i]
        print(" "*(i+1), "node {} ".format(i), c.go_to, " fail:", c.fail_transition_state, " ", c.output)
    print("res: ", ac.search("abccab"))
