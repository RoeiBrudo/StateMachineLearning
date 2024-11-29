from helpers import *


class NodeTree:
    def __init__(self, str = "", parent=None):
        self.str = str
        self.children = []
        self.parent=parent

    def add_children(self, str=""):
        child = NodeTree(str=str, parent=self)
        self.children.append(child)
        return child

    def print(self):
        print("str = ", self.str, "my childs", [c.str for c in self.children])
        for son in self.children:
            son.print()

    def get_place(self):
        if self.parent is not None:
            return self.parent.children.index(self)


def sift(ORC, node, s):
    cur = node
    while len(cur.children) != 0:
        d = cur.str
        if d == 'E': d = ''
        sd = s + d
        result = ORC.member_query(str2arr(sd))
        if result: cur = cur.children[0]
        else: cur = cur.children[1]

    return cur.str


def update_tree(ORC, root, counter, hyp):
    s = []
    s_t = []
    pre_strs = [arr2str(counter[:i]) for i in range(1,len(counter)+1)]
    for pre in pre_strs:
        s.append(sift(ORC, root, arr2str(pre)))
        s_t.append(hyp.predict(pre, predict_state=True))

    j = 0
    for i in range(1, len(counter)+1):
        if s[i] != s_t[i]:
            j = i
            break

    s_j = s[j]
    s_t_j = s_t[j]
    # print("sj", s_j, "stj",  s_t_j)
    d = find_dist(s_j, s_t_j, root)
    gamma_j = str(counter[j])
    dist_d = gamma_j + d


    min_pre = pre_strs[j-1]
    s_j_m1 = s[j-1]

    is_found, node_to_replace = find_in_tree(s_j_m1, root)
    replace_node(ORC, node_to_replace, min_pre, s_j_m1, dist_d)


def replace_node(ORC, node, str_new, str_old, d):
    new_node = NodeTree(str=d, parent=node.parent)
    is_right = ORC.member_query(str2arr(str_new+d))
    if is_right:
        new_node.add_children(str_new)
        new_node.add_children(str_old)
    else:
        new_node.add_children(str_old)
        new_node.add_children(str_new)

    node_place = node.get_place()
    node.parent.children[node_place] = new_node


def find_dist(s1, s2, start):
    cur = start
    in_left_1, node_l_1 = find_in_tree(s1, cur.children[0])
    in_right_1, node_r_1 = find_in_tree(s1, cur.children[1])
    in_left_2, node_l_2 = find_in_tree(s2, cur.children[0])
    in_right_2, node_r_2 = find_in_tree(s2, cur.children[1])

    if (in_left_1 and in_right_2) or (in_left_2 and in_right_1):
        return cur.str

    if in_left_1 and in_left_2:
        return find_dist(s1, s2, cur.children[0])

    if in_right_1 and in_right_2:
        return find_dist(s1, s2, cur.children[1])

    else:
        print("what is going all")
        return None

def find_in_tree(str, start):
    nodes = [start]
    while len(nodes) != 0:
        cur = nodes[0]
        if cur.str != str:
            nodes.extend(cur.children)
            nodes.remove(cur)
        else:
            if len(cur.children) == 0:
                return True, cur
            else:
                nodes.extend(cur.children)
                nodes.remove(cur)
    return False, False


def get_leaves(node):
    if len(node.children) == 0:
        yield node

    for child in node.children:
        for leaf in get_leaves(child):
            yield leaf
