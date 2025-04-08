import fitz  # PyMuPDF
import cv2
import numpy as np

def find_osha_symbol_in_pdf(pdf_path, symbol_image_path, threshold=0.8):
    """
    Scans a PDF document for image objects, compares them to a given OSHA
    hazard symbol image, and returns True if a match is found.

    Args:
        pdf_path (str): Path to the PDF file.
        symbol_image_path (str): Path to the PNG image of the OSHA symbol.
        threshold (float): Similarity threshold for template matching (0-1).

    Returns:
        bool: True if the OSHA symbol is found, False otherwise.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        template = cv2.imread(symbol_image_path, cv2.IMREAD_GRAYSCALE)

        if template is None:
            print(f"Error: Could not load template image: {symbol_image_path}")
            return False

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                if base_image is None:
                    continue #skip to next image if extract_image fails.

                image_bytes = base_image["image"]
                image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue #skip if image decode fails.

                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > threshold:
                    print(f"OSHA symbol found on page {page_num + 1}, image index {img_index}")
                    pdf_document.close()
                    return True

        pdf_document.close()
        return False

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False

# Example Usage:
pdf_file = "/home/cythreal/workspace/github.com/cythreal/OIR/hydrogen_peroxide.pdf"  # Replace with your PDF file path
osha_symbol_image = "/home/cythreal/workspace/github.com/cythreal/OIR/ex_mark.png"  # Replace with the OSHA symbol image path

if find_osha_symbol_in_pdf(pdf_file, osha_symbol_image):
    print("OSHA symbol found in PDF.")
else:
    print("OSHA symbol not found in PDF.")