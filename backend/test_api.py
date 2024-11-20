# test_api.py
import requests
import json
import time
import subprocess
import sys
import signal
from typing import Optional
import psutil
import threading
from pathlib import Path


class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.server_process: Optional[subprocess.Popen] = None

    def _stream_output(self, pipe, prefix: str):
        """Stream output from the server process"""
        try:
            for line in pipe:
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                print(f"{prefix}: {line.strip()}")
        except Exception as e:
            print(f"Error in output streaming: {e}")

    def start_server(self):
        """Start the FastAPI server with output streaming"""
        print("Starting FastAPI server...")
        try:
            # First, kill any existing processes on port 8000
            self._kill_existing_server()

            # Start the server
            self.server_process = subprocess.Popen(
                ["uvicorn", "app.main:app", "--reload", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True  # This handles the decoding for us
            )

            # Start output streaming threads
            stdout_thread = threading.Thread(
                target=self._stream_output,
                args=(self.server_process.stdout, "SERVER"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._stream_output,
                args=(self.server_process.stderr, "ERROR"),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait briefly for startup
            time.sleep(3)

            if self.server_process.poll() is not None:
                print("Server process terminated unexpectedly!")
                return False

            print("Server process started")
            return True

        except Exception as e:
            print(f"Error starting server: {e}")
            return False

    def _kill_existing_server(self):
        """Kill any existing process on port 8000"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr.port == 8000:
                            print(
                                f"Killing existing process on port 8000 (PID: {proc.pid})")
                            psutil.Process(proc.pid).terminate()
                            time.sleep(1)  # Wait for termination
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error killing existing server: {e}")

    def stop_server(self):
        """Stop the FastAPI server"""
        if self.server_process:
            print("Stopping server...")
            try:
                if sys.platform == 'win32':
                    self.server_process.kill()
                else:
                    pgid = os.getpgid(self.server_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(1)
                    if self.server_process.poll() is None:  # If still running
                        os.killpg(pgid, signal.SIGKILL)
            except Exception as e:
                print(f"Error stopping server: {e}")
                # Force kill as last resort
                try:
                    self.server_process.kill()
                except:
                    pass
            finally:
                print("Server stopped!")

    def wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready"""
        print(f"Waiting up to {timeout} seconds for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                if self.server_process.poll() is not None:
                    print("Server process has terminated!")
                    return False
                print("Waiting for server to be ready...")
                time.sleep(1)
        return False

    def test_endpoints(self):
        """Test all endpoints"""
        try:
            # Test health endpoint first
            print("\nTesting health endpoint...")
            response = requests.get(f"{self.base_url}/health")
            print(json.dumps(response.json(), indent=2))

            # Test root endpoint
            print("\nTesting root endpoint...")
            response = requests.get(self.base_url)
            print(json.dumps(response.json(), indent=2))

            # Test prediction endpoint
            print("\nTesting prediction endpoint...")
            test_data = {
                "points": [[100, 100], [120, 120], [140, 140]]
            }
            response = requests.post(
                f"{self.base_url}/api/v1/predict",
                json=test_data
            )
            print(json.dumps(response.json(), indent=2))

            return True

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to server. Make sure it's running!")
            return False
        except Exception as e:
            print(f"Error during testing: {e}")
            return False


def verify_environment():
    """Verify all required files and directories exist"""
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/model/__init__.py",
        "app/model/cnn.py",
        "app/utils/__init__.py",
        "app/utils/preprocessing.py"
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            print(f"Missing required file: {file_path}")
            all_exist = False

    if all_exist:
        print("Environment verification passed!")
    return all_exist


def main():
    print("Starting tests...")

    # Verify environment first
    if not verify_environment():
        print("Environment verification failed!")
        return

    tester = APITester()

    try:
        # Start server
        if not tester.start_server():
            print("Failed to start server!")
            return

        # Wait for server to be ready
        if not tester.wait_for_server(timeout=15):
            print("Error: Server failed to start!")
            return

        # Run tests
        tester.test_endpoints()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        tester.stop_server()


if __name__ == "__main__":
    main()
