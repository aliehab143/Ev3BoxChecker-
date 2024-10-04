
# **EV3 Box Checker Project**
![LEGO Logo](https://logos-world.net/wp-content/uploads/2020/09/LEGO-Symbol.png)

**EV3 Box Checker** is a Flutter-based mobile application designed to automate the process of verifying the contents of a LEGO Mindstorms EV3 kit using AI and image processing. The app integrates with a backend service to analyze images and identify whether all the components of the kit are present.

## **Project Overview**

The **EV3 Box Checker** is aimed at helping educators, students, and hobbyists ensure that their LEGO Mindstorms EV3 kits contain all necessary pieces before or after a session. Using AI-driven image segmentation and object recognition, the app provides an easy, automated way to track missing or misplaced components in real-time.

## **Features**

- **Image Upload**: Users can take or upload images of their LEGO EV3 kits.
- **Automated Component Recognition**: Using an integrated AI model, the app identifies and classifies EV3 components.
- **Real-Time Analysis**: Get immediate feedback on which components are missing or incorrectly placed.
- **User-Friendly Interface**: Intuitive Flutter-based design for easy navigation and interaction.

## **Architecture**

The architecture consists of several key components:
- **Flutter App**: The client-side application used by users to interact with the system, upload images, and view results.
- **Ngrok Tunnel**: A public tunnel exposing the local backend service to the internet.
- **Python Backend**: A Flask-based backend that handles image uploads, sends data to the segmentation model, and returns results.
- **Segmentation Model**: An AI model responsible for image segmentation, identifying LEGO components in the images.
- **Brickognize Service**: A specialized service that verifies the identified components based on pre-defined criteria.

## **Demo Video**

Click the image below to watch the demo video:

[![Demo Video](https://img.youtube.com/vi/-l1dDFbryz4/0.jpg)](https://youtu.be/-l1dDFbryz4)


## **Technologies Used**

- **Frontend**: Flutter (Dart)
- **Backend**: Python (Flask)
- **Machine Learning**: Image Segmentation Model for object detection
- **Tunneling**: Ngrok for exposing the backend to the internet
- **Storage**: Local storage on the client-side for caching

## **Getting Started**

### **Prerequisites**
To run the project, you will need:
- [Flutter SDK](https://flutter.dev/docs/get-started/install)
- [Python 3.x](https://www.python.org/)
- [Ngrok](https://ngrok.com/)
- Flask and related Python libraries (detailed in `requirements.txt`)

### **Installation Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aliehab143/Ev3BoxChecker-.git
   ```
   
2. **Set Up the Flutter App**:
   Navigate to the `ev3-box-checker` folder and run:
   ```bash
   flutter pub get
   flutter run
   ```

3. **Set Up the Python Backend**:
   Navigate to the backend directory and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Run the Flask server:
   ```bash
   python app.py
   ```

4. **Run Ngrok**:
   In a separate terminal, run:
   ```bash
   ngrok http 5000
   ```
   This will expose the Flask server on a public URL that you can use in your Flutter app.

### **Usage**

- Launch the Flutter app on a mobile device or emulator.
- Upload or take a picture of your EV3 kit using the app.
- The image will be sent to the backend for analysis, and you will receive a list of identified components along with missing items.

## **Contributing**

Contributions are welcome! Feel free to submit pull requests to suggest improvements or add new features.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push your changes and create a pull request.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contact**

For any questions or support regarding this project, please feel free to reach out to the repository maintainer via [GitHub Issues](https://github.com/aliehab143/Ev3BoxChecker-/issues).

---
