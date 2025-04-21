🔧 How to Run the Project
This Satellite Image Classification project is built using Python, TensorFlow/Keras, and Tkinter, and is designed to be run locally on your system. Please follow the steps below to set up and run the project smoothly.

✅ Prerequisites
Before running the project, make sure you have the following installed:

Python 3.7 or later

pip (Python package installer)

Required Python libraries:

tensorflow

numpy

opencv-python

pillow

tkinter (comes pre-installed with Python)

pytesseract (for OCR)

Tesseract OCR Engine (external dependency)

📁 Project Setup
Clone or Download the Repository

bash
Copy
Edit
git clone https://github.com/your-username/satellite-image-classification.git
cd satellite-image-classification
Install Required Libraries

Run the following command to install the necessary Python packages:

bash
Copy
Edit
pip install -r requirements.txt
(If requirements.txt is not available, manually install the dependencies using pip install tensorflow opencv-python pillow pytesseract)

Install Tesseract-OCR

Download and install Tesseract from https://github.com/tesseract-ocr/tesseract

After installation, make sure the Tesseract path is added to your system's environment variables.

For Windows users, typically:

makefile
Copy
Edit
C:\Program Files\Tesseract-OCR\tesseract.exe
Place the Model File

Make sure the pre-trained model file (student_model.h5) is placed in the correct directory as referenced in the Python script. Update the path in the code if needed.

▶️ Run the Application
Simply run the main Python script:

bash
Copy
Edit
python gui_app.py
This will launch the GUI window where you can upload a satellite image and see the classification result.

⚠️ Note on Image Validity
The system automatically detects:

Human faces

Printed or handwritten text

If either is found in the uploaded image, it will notify the user and reject the image as it is not a valid satellite input. Please ensure you upload real satellite images for accurate classification.

🔒 Copyright Notice
This project is a copyrighted work developed by Mahesh Bhakare.
Unauthorized reproduction, redistribution, or commercial use is strictly prohibited.

Use of this project, in whole or in part, is only permitted with prior written consent from the author.

If you wish to use, collaborate, or reference this project for academic, research, or personal purposes, kindly reach out to the author:

📞 Contact: +91 8390704252
📧 Email: msbhakare5@gmail.com

