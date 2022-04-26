from utility.logger_decor import exception


@exception
def main():
    a = 1/0
    print("main")


if __name__ == '__main__':
    main()
