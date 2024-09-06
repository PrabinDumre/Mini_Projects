import os

if __name__ == '__main__':
    print("Welcome to Robot Speaker." )
    print( "Type 'exit' to stop")
    os.system("say 'Hello, I am your voice. You can type words you want to spell and I will do it for you '")
    while(True):

        x = input("Enter what you want me to Speak : ")
        if x == "exit":
            os.system("say 'Bye Bye Thank you for using me'")
            break
        command = f"say {x}"
        os.system(command)



