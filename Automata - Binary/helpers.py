def str2arr(str):
    if str == "": return []
    arr = []
    for i in str:
        arr.append(int(i))
    return arr


def arr2str(arr):
    if arr == []: return ""
    return ''.join(str(e) for e in arr)
