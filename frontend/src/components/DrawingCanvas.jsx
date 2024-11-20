import React, { useState, useRef } from 'react';
import axios from 'axios';
import { ReactSketchCanvas } from 'react-sketch-canvas';

const DrawingCanvas = () => {
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const canvasRef = useRef(null);

  const handlePredict = async () => {
    if (!canvasRef.current) return;

    setIsLoading(true);
    try {
      // Get paths from canvas
      const paths = await canvasRef.current.exportPaths();
      
      // Convert paths to points array
      const points = paths.flatMap(path => 
        path.paths.map(point => [point.x, point.y])
      );

      // Send to backend
      const response = await axios.post('http://localhost:8000/api/v1/predict', {
        points: points
      });

      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
      setPredictions([]);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-4 text-center">Doodle Recognition</h1>
        
        <div className="border-2 border-gray-200 rounded-lg overflow-hidden mb-4">
          <ReactSketchCanvas
            ref={canvasRef}
            strokeWidth={3}
            strokeColor="black"
            canvasColor="white"
            width="100%"
            height="400px"
            className="touch-none"
          />
        </div>

        <div className="flex gap-4 justify-center mb-4">
          <button
            onClick={handlePredict}
            disabled={isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                     disabled:bg-blue-400 disabled:cursor-not-allowed
                     transition-colors duration-200"
          >
            {isLoading ? 'Predicting...' : 'Predict'}
          </button>
          <button
            onClick={handleClear}
            className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 
                     transition-colors duration-200"
          >
            Clear
          </button>
        </div>

        {predictions.length > 0 && (
          <div className="border-t pt-4">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Predictions
            </h2>
            <ul className="space-y-3">
              {predictions.map(([label, confidence], index) => (
                <li 
                  key={index}
                  className="flex justify-between items-center bg-gray-50 p-3 rounded-lg"
                >
                  <span className="font-medium text-gray-900 capitalize">
                    {label}
                  </span>
                  <span className="text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default DrawingCanvas;