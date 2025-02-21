Creating a complete Python program for a "smart-garden" system involving IoT sensors and machine learning is a complex task. Below, I'll outline a simplified version of such a system, focusing on the core components and principles. This simplified version includes a basic machine learning model, data collection via simulated IoT sensors, and an irrigation control system. For a real-world application, you'd need additional features such as hardware integration, advanced machine learning models, and deployment on an IoT platform.

```python
import random
import time
import logging
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

# Setting up logging for debugging and error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartGarden:
    def __init__(self):
        # Initialize simulated sensors
        self.soil_moisture_sensor = 0
        self.light_sensor = 0
        self.temperature_sensor = 0
        self.irrigation_status = False
        
        # Load or train a simple machine learning model
        self.model = self.train_model()
        
    def collect_sensor_data(self):
        """Simulate collecting data from sensors."""
        try:
            self.soil_moisture_sensor = random.uniform(0, 100)  # Percentage
            self.light_sensor = random.uniform(200, 1000)  # Lux
            self.temperature_sensor = random.uniform(10, 40)  # Celsius
            logger.info(f'Sensor data collected: Soil moisture={self.soil_moisture_sensor:.2f}%, '
                        f'Light={self.light_sensor:.2f} Lux, '
                        f'Temperature={self.temperature_sensor:.2f}°C')
        except Exception as e:
            logger.error(f"Error collecting sensor data: {e}")
    
    def train_model(self):
        """Train a simple Decision Tree model to predict if irrigation is needed."""
        try:
            # Example training data and labels (in a realistic scenario, collect real data)
            X_train = np.array([[30, 400, 25], [70, 700, 20], [20, 300, 30], [90, 800, 20]])
            y_train = np.array([1, 0, 1, 0])  # 1: irrigation needed, 0: irrigation not needed

            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            logger.info("Model trained successfully.")
            return model
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def predict_irrigation(self):
        """Make a prediction whether irrigation is needed."""
        try:
            features = np.array([[self.soil_moisture_sensor, self.light_sensor, self.temperature_sensor]])
            prediction = self.model.predict(features)
            logger.info(f'Irrigation prediction: {"Needed" if prediction else "Not needed"}')
            return bool(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return False
    
    def control_irrigation(self):
        """Control the irrigation system based on the prediction."""
        try:
            self.irrigation_status = self.predict_irrigation()
            if self.irrigation_status:
                self.start_irrigation()
            else:
                self.stop_irrigation()
        except Exception as e:
            logger.error(f"Error controlling irrigation: {e}")
    
    def start_irrigation(self):
        """Start the irrigation system."""
        logger.info("Irrigation started.")
        # Simulate irrigation logic
        
    def stop_irrigation(self):
        """Stop the irrigation system."""
        logger.info("Irrigation stopped.")
        # Simulate stopping irrigation logic

def main():
    garden = SmartGarden()
    while True:
        garden.collect_sensor_data()
        garden.control_irrigation()
        time.sleep(5)  # Wait for 5 seconds before the next cycle

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Smart garden system terminated manually.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
```

### Explanation:
- **Simulated Sensors:** The program includes simulated soil moisture, light, and temperature sensors, since full IoT integration would require hardware and connectivity.
- **Machine Learning Model:** A simple Decision Tree Classifier is used to predict whether irrigation is needed based on the sensor readings. This is a placeholder for more sophisticated models that might be trained with real data.
- **Irrigation Control:** The program decides whether to irrigate based on model predictions.
- **Error Handling:** Logging and try-except blocks are implemented for debugging and error management.
- **Execution Loop:** Continuously collects sensor data and manages irrigation in a loop, emulating real-time operation.

This is a simplified example for educational purposes. Advanced implementations would require thorough consideration of sensor data accuracy, real-world conditions, and specific plant requirements for optimization.