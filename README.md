# rag-budget-chatbot
# README

## Project Overview
This project serves as a web-based and Flask-powered application for document processing and search functionalities.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies (install using `pip install -r requirements.txt`)

## How to Run the `index.html` File
1. Open `index.html` in a web browser directly, or
2. Use a simple HTTP server to serve the file:
   ```sh
   python -m http.server 8000
   ```
   Then, open `http://localhost:8000/index.html` in a web browser.
3. Alternatively, if using Visual Studio Code:
   - Install the **Live Server** extension.
   - Right-click on `index.html` and select **Open with Live Server**.
   - This will automatically launch the file in a browser with live reloading.

## How to Run `app.py`
1. Ensure all dependencies are installed.
2. Run the Flask app using:
   ```sh
   python app.py
   ```
3. The app should now be accessible at `http://127.0.0.1:5000/` in your browser.

## Troubleshooting
- If you encounter missing module errors, install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Ensure the Flask app is running before accessing it in the browser.

## License
This project is licensed under the MIT License.

