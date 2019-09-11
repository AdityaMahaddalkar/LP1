import aiml
import os

def chat():
    ''' 
    Function for continuous chat loop
    Input: void
    Output: return value 0 if success 1 if failure
    '''
    message = ''
    kernel = aiml.Kernel()
    '''
    if os.path.isfile('../resources/bot_brain.brn'):
        print('======Bootstraping=======')
        kernel.bootstrap(brainFile='../resources/bot_brain.brn')
    else:
        kernel.bootstrap(learnFiles=os.path.abspath('../resources/std-startup.xml'), commands="load aiml b")
        kernel.saveBrain("../resources/bot_brain.brn")
    '''

    kernel.learn("basic_chat.aiml")
    try:
        while True:
            message = input('>')

            if message == 'exit':
                break
            elif message == 'save':
                kernel.saveBrain('../resources/bot_brain.brn')
            else:
                bot_response = kernel.respond(message)
                print(bot_response)
    
    except Exception as e:
        print(e)
        return 1
    return 0


if __name__ == '__main__':
	chat()