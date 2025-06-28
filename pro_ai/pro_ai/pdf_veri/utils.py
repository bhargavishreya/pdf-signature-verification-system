# import fitz  
# import cv2
# import numpy as np
# from PIL import Image
# import io
# from ultralytics import YOLO

# def verify_pdf_signature_and_stamp(pdf_file):

#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     page = doc.load_page(0)
#     pix = page.get_pixmap(dpi=300)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     doc.close()


#     image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


#     height, width = image.shape[:2]
#     top = int(height * 0.78)
#     bottom = height
#     left = int(width * 0.60)
#     right = width
#     roi = image[top:bottom, left:right]

   
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
          
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 
#     significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # 50 px area threshold
  


#     has_mark = len(significant_contours) > 0

#     return has_mark, has_mark

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io

def verify_pdf_contents(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect outermost box (largest contour)
    _, thresh_outer = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, ["No outer box detected."]

    outer_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(outer_contour)
    cropped = image[y:y+h, x:x+w]

    # Step 2: OCR for 4M Change/No Change
    text = pytesseract.image_to_string(cropped).lower()
    if "4m change" not in text and "4m no change" not in text:
        return False, ["Missing '4M change' or '4M no change' in the document."]

    # Step 3: Detect blank box (near bottom-right assumed)
    height, width = cropped.shape[:2]
    blank_box_area = cropped[int(height*0.78):height, int(width*0.7):width]
    blank_gray = cv2.cvtColor(blank_box_area, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(blank_gray)
    if mean_intensity < 200:
        return False, ["Blank box beside signature/seal not detected."]

    # Step 4: Detect signature/stamp (middle bottom-right region)
    sig_area = cropped[int(height*0.78):height, int(width*0.4):int(width*0.7)]
    sig_gray = cv2.cvtColor(sig_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(sig_gray, (5, 5), 0)
    _, thresh_sig = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_sig, _ = cv2.findContours(thresh_sig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_sig = [cnt for cnt in contours_sig if cv2.contourArea(cnt) > 80]

    if len(significant_sig) < 2:
        return False, ["Signature and stamp not properly detected."]

    return True, ["All checks passed: Verified."]
