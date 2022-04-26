import sys
sys.path.append("./py_basic/")
from event.send import Sender
from event.push import msg

# print(sys.path)
def main():

    s = Sender()
    s.log("Core > Event > Send")

    msg.output()

# main()


def main():

    txt = open("./data/fradulent_emails.txt")
    info = txt.fileno()
    print(info)
    txt.close()
    

if __name__ == '__main__':
    main()
