from automata_extractor.automata.AngluinClassifier import helpers as h


class NodeTree:
    def __init__(self, classes, str, is_generic=False, parent=None):
        self.str = str
        self.classes = classes
        self.parent = parent
        self.children = {}
        self.is_generic = is_generic

    def add_children(self, child_class, str, is_generic=False):
        child = NodeTree(self.classes, str=str, is_generic=is_generic, parent=self)
        self.children[child_class] = child
        return child

    def add_generic(self, g):
        for i,c in enumerate(self.classes):
            if c not in self.children.keys():
                s = 'GenericNode {},{}'.format(g, i)
                self.add_children(c, str=s, is_generic=True)

    def is_leaf(self):
        return not self.children.keys()

    def print(self):
        if self.str == '':
            node_str = 'E'
        else:
            node_str = self.str
        print("name = ", node_str, "my-childs: ", [self.children[key].str for key in sorted(self.children.keys())])
        for key in sorted(self.children.keys()):
            self.children[key].print()

    def get_class(self):
        for key in self.parent.children.keys():
            if self.parent.children[key] is self:
                return key

    def __lt__(self, other):
        return self.str < other.str


def sift(ORC, node, s):
    cur = node
    while len(cur.children) != 0:
        d = cur.str
        sd = h.add_strings(s, d)
        result = ORC.member_query(h.key2word(sd))
        cur = cur.children[result]

    if cur.is_generic:
        cur.is_generic = False
        cur.str = s

    return cur


def update_tree(ORC, root, counter, hyp, g):
    s = []
    s_t = []
    pre_strs = h.get_all_prefix(counter)
    j = -1
    i = -1
    for pre in pre_strs:
        i += 1
        s_i = sift(ORC, root, pre).str
        s_t_i = hyp.predict(h.key2word(pre), predict_state=True)
        s.append(s_i)
        s_t.append(s_t_i)
        if s_i != s_t_i:
            j = i
            break

    if j == -1:
        print("Error!!!!!")
        return

    s_j = s[j]
    s_t_j = s_t[j]
    d = find_dist(s_j, s_t_j, root)

    gamma_j = pre_strs[j].split(',')[-1]
    dist_d = h.add_strings(gamma_j, d)
    min_pre = pre_strs[j-1]
    s_j_m1 = s[j-1]

    is_found, node_to_replace = find_in_tree(s_j_m1, root)
    replace_node(ORC, node_to_replace, min_pre, s_j_m1, dist_d, g)
    # print("end updating")


def replace_node(ORC, node, str_new, str_old, d, g):
    new_node = NodeTree(str=d, parent=node.parent, is_generic=False, classes=node.classes)
    class_new = ORC.member_query(h.key2word(h.add_strings(str_new, d)))
    class_old = ORC.member_query(h.key2word(h.add_strings(str_old, d)))
    new_node.add_children(class_new, str=str_new)
    new_node.add_children(class_old, str=str_old)
    new_node.add_generic(g)

    node_class = node.get_class()
    node.parent.children[node_class] = new_node


def find_dist(s1, s2, start):
    cur = start
    found_1 = False
    found_2 = False
    for key in cur.children.keys():
        is_1_in_key_kid, drop = find_in_tree(s1, cur.children[key])
        is_2_in_key_kid, drop = find_in_tree(s2, cur.children[key])

        if is_1_in_key_kid:
            found_1 = True

        if is_2_in_key_kid:
            found_2 = True

        if is_1_in_key_kid and is_2_in_key_kid:
            return find_dist(s1, s2, cur.children[key])

    if not found_1 or not found_2:
        print("Error dist!")

    else:
        return cur.str


def find_in_tree(str, start):
    nodes = [start]
    while len(nodes) != 0:
        cur = nodes[0]
        if cur.str != str:
            nodes.extend(cur.children.values())
            nodes.remove(cur)
        else:
            if len(cur.children.keys()) == 0:
                return True, cur
            else:
                nodes.extend(cur.children.values())
                nodes.remove(cur)
    return False, False


def get_leaves(node):
    if len(node.children.keys()) == 0:
        yield node

    for child in node.children.keys():
        for leaf in get_leaves(node.children[child]):
            yield leaf
