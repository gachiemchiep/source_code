class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

# Write a function revstring(mystr) that uses a stack to reverse the characters in a string.
def  revstring(mystr:str) -> str:

    s = Stack()
    # put into stack
    for char in mystr:
        s.push(char)
    # pop again to form a reverse string
    new_string = ""
    while not s.isEmpty():
        new_string+= s.pop()

    return new_string

# Simple Balanced Parentheses : 
def parChecker(symbolString:str) -> bool:
    s = Stack()
    is_balanced = True
    index = 0

    while (index < len(symbolString)) and is_balanced:
        char = symbolString[index]
        if char == "(":
            s.push(char)
        elif char == ")":
            if s.isEmpty():
                is_balanced = False
            else:
                s.pop()
        index += 1

    return is_balanced

# Simple Balanced Parentheses : [{()}]
def parChecker2(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in "([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top,symbol):
                       balanced = False
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)

# Converting Decimal Numbers to Binary Numbers
def baseConverter(decNumber, base=2):
    digits = "0123456789ABCDEF"
    remstack = Stack()

    while decNumber > 0:
        rem = decNumber % base
        remstack.push(rem)
        decNumber = decNumber // base

    binString = ""
    while not remstack.isEmpty():
        binString = binString + digits[remstack.pop()]

    return binString

if __name__ == '__main__':
    s=Stack()
    print(s.isEmpty())
    s.push(4)
    s.push('dog')
    print(s.peek())
    s.push(True)
    print(s.size())
    print(s.isEmpty())
    s.push(8.4)
    print(s.pop())
    print(s.pop())
    print(s.size())

    print(parChecker('((()))'))
    print(parChecker('(()'))

    print(baseConverter(256, 16))