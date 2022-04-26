def do_foo():  
    print("foo!")  
  
def do_bar():  
    print("bar!")  
  
class Print():  
    def do_foo(self):  
        print("foo!")  
  
    def do_bar(self):  
        print("bar!")  
 
    @staticmethod  
    def static_foo():  
        print("static foo!")  
 
    @staticmethod  
    def static_bar():  
        print("static bar!")


if __name__ == '__main__':
    obj = Print()  
    func_name = "do_foo"  
    eval(func_name)()
    # getattr(obj, func_name)()
    # 
    c_name="Print"
    c=eval(c_name)
    c=c()
    getattr(c, func_name)()