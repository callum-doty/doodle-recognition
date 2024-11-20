# Doodle Recognition App

A web application that recognizes hand-drawn doodles using a Convolutional Neural Network (CNN).

## Features

- Real-time doodle recognition
- Interactive drawing canvas
- Top-3 predictions with confidence scores
- Backend API with FastAPI
- Frontend with React and TailwindCSS

## Tech Stack

### Backend

- Python 3.9+
- FastAPI
- PyTorch
- NumPy
- OpenCV

### Frontend

- React
- TailwindCSS
- React Sketch Canvas
- Axios

## Project Structure

```
doodle-recognition/
├── backend/
│   ├── app/
│   │   ├── model/
│   │   │   └── cnn.py
│   │   ├── utils/
│   │   │   └── preprocessing.py
│   │   └── main.py
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   └── models/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── DrawingCanvas.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── public/
└── README.md
```

## Setup

### Backend

1. Create and activate virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python improved_train.py
```

4. Start the server:

```bash
uvicorn app.main:app --reload
```

### Frontend

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

## Usage

1. Open `http://localhost:5173` in your browser
2. Draw a doodle in the canvas
3. Click "Predict" to see the recognition results
4. Use "Clear" to start over

## Training Data

The model is trained on the [Quick Draw Dataset](https://quickdraw.withgoogle.com/data) and includes the following categories:

- apple
- banana
- cat
- dog
- elephant
- fish
- guitar
- house
- lion
- pencil
- pizza
- rabbit
- snake
- spider
- tree

## Model Architecture

The CNN architecture consists of:

- 3 convolutional layers with batch normalization
- MaxPooling layers
- Dropout for regularization
- 2 fully connected layers
- Output layer with 15 classes

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
