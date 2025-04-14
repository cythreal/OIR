import fitz  # PyMuPDF
import cv2
import numpy as np
import os

def find_images_in_pdfs(pdf_folder, image_folder, threshold=0.8):
    """
    Scans multiple PDF documents in a folder for images located in another folder.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        image_folder (str): Path to the folder containing PNG images to search for.
        threshold (float): Similarity threshold for template matching (0-1).

    Returns:
        dict: A dictionary where keys are PDF file names and values are lists of found image indices.
    """

    found_images = {}

    try:
        image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.lower().endswith(".png")]

        if not image_paths:
            print("Error: No PNG images found in the specified image folder.")
            return {}

        if len(image_paths) > 9: #change if more than 9 hazards
            print("Error: Maximum of 9 images allowed.")
            return {}

        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, filename)
                found_images[filename] = []  # Initialize list for this PDF

                pdf_document = fitz.open(pdf_path)
                templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

                if any(template is None for template in templates):
                    print(f"Error: Could not load one or more template images for {filename}.")
                    continue #skip this pdf.

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

                        for template_index, template in enumerate(templates):
                            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                            if max_val > threshold:
                                found_images[filename].append((page_num + 1, img_index, template_index))
                                #add page number, image index and template index to the list.
                pdf_document.close()

    except Exception as e:
        print(f"Error processing PDFs: {e}")

    return found_images

# Example Usage:
pdf_folder = "/home/cythreal/workspace/github.com/cythreal/OIR/pdf_folder"  # Replace with your PDF folder path
image_folder = "/home/cythreal/workspace/github.com/cythreal/OIR/png_folder" #replace with your image folder path.

found_results = find_images_in_pdfs(pdf_folder, image_folder)

for pdf_name, results in found_results.items():
    if results:
        print(f"Images found in {pdf_name}:")
        for page, image_index, template_index in results:
            print(f"  Page {page}, Image Index: {image_index}, Template index: {template_index}, Template: {os.listdir(image_folder)[template_index]}")
    else:
        print(f"No images found in {pdf_name}.")