
import fitz 
import cv2
import numpy as np
from PIL import Image
import pytesseract
from django.shortcuts import render
from django.http import JsonResponse
from .forms import PDFUploadForm

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def home(request):
    return render(request, 'home.html')


def success_page(request):
    return render(request, 'pdf_veri/success.html')


def verify_pdf_no_change(request):
    return handle_pdf_verification(request, change_type="NO-CHANGE")


def verify_pdf_change(request):
    return handle_pdf_verification(request, change_type="CHANGE")


def handle_pdf_verification(request, change_type):
    result = None
    errors = []

    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf_file']

            if not pdf_file.name.lower().endswith('.pdf'):
                errors.append("❌ File must be a PDF.")
            elif pdf_file.size > 10 * 1024 * 1024:
                errors.append("❌ PDF must be under 10 MB.")
            else:
                try:
                    pdf_file.seek(0)
                    results = verify_pdf_full_pipeline(pdf_file)
                    if results['final_result']:
                        result = results['reason'][0]
                    else:
                        errors.extend(results['reason'])
                except Exception as e:
                    errors.append(f"❌ Error while verifying PDF: {str(e)}")
        else:
            errors.append("❌ Invalid form.")

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'success': not errors,
                'form_valid': form.is_valid(),
                'errors': errors,
                'result': result,
                'change_type': change_type
            })

    else:
        form = PDFUploadForm()

    return render(request, 'verify_pdf.html', {
        'form': form,
        'result': result,
        'errors': errors,
        'change_type': change_type
    })


def verify_pdf_full_pipeline(pdf_file):
    results = {
        "text_found": False,
        "blank_box_found": False,
        "signature_found": False,
        "stamp_found": False,
        "final_result": False,
        "reason": []
    }

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    image = cv2.cvtColor(
        np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)),
        cv2.COLOR_RGB2BGR
    )
    doc.close()

    height, width = image.shape[:2]

    # Step 1: Find outermost box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        outer_box = image[y:y + h, x:x + w]
    else:
        results['reason'].append("❌ Outer box not found.")
        return results

    # Step 2: Detect text
    top_roi = outer_box[0:int(h * 0.3), :]
    text = pytesseract.image_to_string(top_roi)
    if "4M change" in text or "4M no change" in text:
        results["text_found"] = True
    else:
        results['reason'].append("❌ '4M change' or '4M no change' text not found.")

    # Step 3 & 4: Analyze signature/stamp section
    bottom_roi = outer_box[int(h * 0.75):, int(w * 0.6):]
    gray_bottom = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_bottom = cv2.threshold(gray_bottom, 180, 255, cv2.THRESH_BINARY_INV)
    contours_bottom, _ = cv2.findContours(thresh_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_bottom:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        box = thresh_bottom[y:y + bh, x:x + bw]
        ratio = cv2.countNonZero(box) / (box.shape[0] * box.shape[1])

        if ratio < 0.05:
            results['blank_box_found'] = True
        elif ratio > 0.3:
            if x < bottom_roi.shape[1] // 2:
                results['signature_found'] = True
            else:
                results['stamp_found'] = True

    if not results['blank_box_found']:
        results['reason'].append("❌ Blank box not detected.")
    if not results['signature_found']:
        results['reason'].append("❌ Signature not detected.")
    if not results['stamp_found']:
        results['reason'].append("❌ Stamp not detected.")

    if all([results['text_found'], results['blank_box_found'], results['signature_found'], results['stamp_found']]):
        results['final_result'] = True
        results['reason'] = ["✅ PDF verified successfully."]

    return results
