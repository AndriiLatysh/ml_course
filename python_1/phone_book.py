class PhoneBook:

    def __init__(self):
        self.phone_book = {}

    def add(self, name, number):
        self.phone_book[name] = number

    def find(self, name):
        if self.contains(name):
            return self.phone_book[name]
        else:
            return "Not found"

    def remove(self, name):
        self.phone_book.pop(name)

    def contains(self, name):
        return name in self.phone_book


