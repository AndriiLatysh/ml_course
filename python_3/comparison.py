import stub_class
import copy

# print(1 == 1.0)
#
# a = 1 / 3
# b = 0.1 / 0.3
#
# print(a == b)

# print("{:.60f}".format(0.1))

# print("abc" < "abd")

n = 1000000
stab_instances = []
stab_instances_ids = []

for z in range(n):
    stab_instance = stub_class.StubClass()
    # print(stab_instance.id, id(stab_instance))
    stab_instances.append(stab_instance)
    stab_instances_ids.append(id(stab_instance))

# print(stab_instances_ids)
stab_instances_ids.sort()

for z in range(1, n):
    if stab_instances_ids[z - 1] == stab_instances_ids[z]:
        print("Wait, what?")

for z in range(n):
    pass

# list_1 = [1, 2, 3]
# list_2 = [1, 2, 3]
# print(list_1 is list_2)
# print(list_1 == list_2)
#
# a = [1, 2, 3]
# b = a
# b[1] = 0
# print(a)
#
# import copy
# a = [1, 2, 3]
# b = copy.deepcopy(a)
# b[1] = 0
# print(a)
