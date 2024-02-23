from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print(pytesseract.image_to_string(Image.open('./data/frame0.jpg')))
print("\n\n\n\nOne Ended Here\n\n\n\n")
print(pytesseract.image_to_string(Image.open('./data/frame1.jpg')))
