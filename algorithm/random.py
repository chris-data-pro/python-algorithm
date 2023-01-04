import time

if __name__ == '__main__':
    def rand(input_int):
        random = int(time.time()*1000)
        return random % input_int

    print(rand(5))
