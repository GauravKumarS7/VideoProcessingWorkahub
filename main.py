# Importing all necessary libraries
import cv2
# import os
# from PIL import Image
from langchain.chat_models import ChatOpenAI
# from langchain.sql_database import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
OPENAI_API_KEY = "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"
# Read the video from specified path
cam = cv2.VideoCapture("https://workahub.s3.amazonaws.com/goappdata/f784b749-a790-4c57-982e-4989c6926079/1709632112/hVrt13P6i5/1709632112.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVMVHX4KRP64G6B6Y%2F20240305%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240305T100316Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=879ff6dc8471ce21f540c15ed4536be61a667cbb41d1dd22a121c44b57bd1975")

# try:

#     # creating a folder named data
#     if not os.path.exists('data'):
#         os.makedirs('data')

# # if not created then raise error
# except OSError:
#     print('Error: Creating directory of data')

convLLM = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1,
                     model_name="gpt-3.5-turbo-1106")
mem = ConversationBufferMemory()
# frame
currentframe = 0
diffscreens = set()
while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        # name = './data/frame' + str(currentframe) + '.jpg'
        yval = int(frame.shape[0]/6)
        xval = int(frame.shape[1]/6)
        requiredframe = frame[yval:int(
            frame.shape[0])-yval, xval:int(frame.shape[1])-xval]
        extractedtext = pytesseract.image_to_string(requiredframe, lang="eng")
        prevsize = len(diffscreens)
        diffscreens.add(extractedtext)

        # writing the extracted images
        # cv2.imwrite(name, frame)

        # show how many frames are created
        cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe*5000))    # move the time
        mem.chat_memory.add_user_message(extractedtext)
        # increasing counter so that it will
        currentframe += 1
    else:
        break

convLlmChain = ConversationChain(
    llm=convLLM, memory=mem)
# print(convLlmChain.run(
#     "from the conversation above analyze and provide a approximate percentage of repeatitions, keeping in check that this data is from user screenshots, also the data is 100% accurate. Don't include anything except percentage"))
# print(convLlmChain.run("From the above conversations which denotes text from user screen recording in last 10 minutes duration and having keyboard clicks: 202, mouse clicks: 54, and cpu usage of 2% of 2.5GHz 12 cores cpu for the 10 minutes duration, just tell roughly whether user has worked or not?"))
print(convLlmChain.run("From the above prompts which are the screenshots text data of user, how will you divide the prompts data into the segments to understand what work has been done by the user"))
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
