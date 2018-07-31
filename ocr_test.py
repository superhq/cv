from PIL import Image
import pytesseract

path = "ocr.png"

text = pytesseract.image_to_string(Image.open(path), lang='chi_sim')
print(text)
