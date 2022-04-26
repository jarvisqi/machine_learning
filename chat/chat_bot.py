from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer,ListTrainer

def main():
    boot = ChatBot("deepThought")
    boot.set_trainer(ChatterBotCorpusTrainer)
    # 使用中文语料库训练它
    boot.train("chatterbot.corpus.chinese")  # 语料库

    while True:
        user_text = input ("User: ")
        print("AI: ",boot.get_response(user_text))

    
if __name__ == '__main__':
    main()
   
