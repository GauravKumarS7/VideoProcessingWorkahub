# Importing all necessary libraries
import cv2
import os
import json
from skimage.metrics import structural_similarity as compare_ssim
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
cam = cv2.VideoCapture(
    "https://workahub.s3.amazonaws.com/goappdata/f784b749-a790-4c57-982e-4989c6926079/1710128953/5a84jjJkwz/1710128953.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVMVHX4KRP64G6B6Y%2F20240311%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240311T043853Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=009ec706d65c3db8bae6731c12efc71d415d487041ec7b0a32e60e95c53b78e4")

try:
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

convLLM = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1,
                     model_name="gpt-3.5-turbo-1106")
mem = ConversationBufferMemory()
# frame
currentframe = 0
screens = []
diffscreens = set()
val = dict()
while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        # name = './data/frames' + str(currentframe) + '.jpg'
        yval = int(frame.shape[0]/8)
        xval = int(frame.shape[1]/8)
        requiredframe = frame[yval:int(
            frame.shape[0])-yval, xval:int(frame.shape[1])-xval]
        extractedtext = pytesseract.image_to_string(requiredframe, lang="eng")
        diffscreens.add(extractedtext)
        val["frame_"+str(currentframe)] = extractedtext
        screens.append(requiredframe)

        # writing the extracted images
        # cv2.imwrite(name, requiredframe)

        # show how many frames are created
        cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe*5000))    # move the time
        # mem.chat_memory.add_user_message(extractedtext)
        # increasing counter so that it will
        currentframe += 1
    else:
        break


val["mouse_clicks"] = 124
val["keyboard_clicks"] = 302
val["cpu_utilization"] = "2%"
val["cpu"] = "2.5 GHz 12 cores cpu"
val["duration"] = "10 minutes"
val["applications_used"] = ["vscode", "brave browser", "workahub"]
val["browser_tabs"] = [
    {
        "title": "S7-Works/Workahub_web_backend",
        "url": "https://github.com/S7-Works/Workahub_web_backend"
    },
    {
        "url": "http://localhost:3000/projects",
        "title": "Workahub"
    },
    {
        "url": "https://teams.microsoft.com/v2/",
        "title": "Chat | Saumya Garg | Microsoft Teams"
    },
    {
        "url": "https://github.com/S7-Works/Workahub_web_backend/blob/master/manage.p",
        "title": "Workahub_web_backend/manage.py at master · S7-Works/Workahub_web_backend"
    }
]
similarityMat = []
for i in range(0, len(screens)):
    row = []
    for j in range(0, len(screens)):
        img1 = screens[i]
        img2 = screens[j]

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(gray1, gray2, full=True)
        row.append(round(score, 5))
    similarityMat.append(row)
val["similarity_matrix_of_frames"] = similarityMat

for key in val.keys():
    mem.chat_memory.add_user_message(str(key)+": "+str(val[key]))
# print(mem.chat_memory.messages)

convLlmChain = ConversationChain(
    llm=convLLM, memory=mem)
with open('result.json', 'w') as fp:
    json.dump(val, fp)
# print(convLlmChain.run(
#     "from the conversation above analyze and provide a approximate percentage of repeatitions, keeping in check that this data is from user screenshots, also the data is 100% accurate. Don't include anything except percentage"))
# print(convLlmChain.run("From the above conversations which denotes text from user screen recording in last 10 minutes duration and having keyboard clicks: 202, mouse clicks: 54, and cpu usage of 2% of 2.5GHz 12 cores cpu for the 10 minutes duration, just tell roughly whether user has worked or not?"))
# print(convLlmChain.run("From the above prompts which are the details of a specific user, provide an approx efficiency score of the user"))


# after feeding the json give this intermediate prompt
# From the above prompts which are the details of a specific user, provide an approx  productivity inference
# after intermediate prompt we can use this prompt to get the result of productivity
# From the above prompts which are the details of a specific user, provide an approx  productivity score in most shortest way possible



# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
