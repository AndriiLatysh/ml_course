import phone_book


contacts = phone_book.PhoneBook()
while True:
    input_list = input().split()
    command = input_list[0]
    name = input_list[1]
    if command == "insert":
        number = input_list[2]
        contacts.add(name, number)
    elif command == "delete":
        contacts.remove(name)
    elif command == "get":
        print(contacts.find(name))
