counter = 1


class StubClass:

    def __init__(self):
        global counter
        self.id = counter
        counter += 1

