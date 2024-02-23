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
cam = cv2.VideoCapture("https://workahub.s3.amazonaws.com/goappdata/f784b749-a790-4c57-982e-4989c6926079/1708686576/X7tLFJ84qY/1708686576.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVMVHX4KRP64G6B6Y%2F20240223%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240223T123559Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=95b9d25c63b5bd8117c92bd02eeffe4ce678dbab06fe4876c7fcc92df3a2a602")

# try:

#     # creating a folder named data
#     if not os.path.exists('data'):
#         os.makedirs('data')

# # if not created then raise error
# except OSError:
#     print('Error: Creating directory of data')

convLLM = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.1,
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
        name = './data/frame' + str(currentframe) + '.jpg'

        extractedtext = pytesseract.image_to_string(frame)
        prevsize = len(diffscreens)
        diffscreens.add(extractedtext)

        # writing the extracted images
        # cv2.imwrite(name, frame)

        # show how many frames are created
        cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe*5000))    # move the time
        mem.save_context({"input": extractedtext}, {
                         "output": ""})

        # increasing counter so that it will
        currentframe += 1
    else:
        break

convLlmChain = ConversationChain(
    llm=convLLM, memory=mem)
print(currentframe)
print("\n")
print(len(diffscreens))
print("\n")
print(convLlmChain.run(
    "from the conversation above analyze and provide a approximate percentage of repeatitions, keeping in check that this data is from user screenshots, also the data is 100% accurate. Don't include anything except percentage"))

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
