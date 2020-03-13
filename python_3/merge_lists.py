def merge_lists(l1, l2):
    lm = []
    p1, p2 = 0, 0
    while p1 < len(l1) and p2 < len(l2):
        if l1[p1] <= l2[p2]:
            lm.append(l1[p1])
            p1 += 1
        elif l1[p1] > l2[p2]:
            lm.append(l2[p2])
            p2 += 1
    for z in range(p1, len(l1)):
        lm.append(l1[z])
    for z in range(p2, len(l2)):
        lm.append(l2[z])
    return lm


l1 = [1, 4, 7, 8, 10, 12, 13]
l2 = [1, 2, 4, 6, 9, 11, 14, 16, 20]

print(merge_lists(l1, l2))
