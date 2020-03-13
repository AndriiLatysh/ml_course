row_count = int(input("Rows: "))
column_count = int(input("Columns: "))
values = []
for x in range(row_count):
    input_row = [int(v) for v in input().split()]
    values.append(input_row)

for x in range(row_count):
    for y in range(column_count):
        print(values[x][y], end=" ")
    print()

for y in range(column_count):
    for x in range(row_count):
        print(values[x][y], end=" ")
    print()
