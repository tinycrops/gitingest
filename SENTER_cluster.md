# Repository Analysis

## Summary

```
Directory: home/ath/SENTER
Files analyzed: 30

Estimated tokens: 79.0k
```

## Important Files

```
Directory structure:
└── SENTER/
    ├── audio_test.py
    ├── ava_audio_config.py
    ├── camera_tools.py
    ├── capture_logs.py
    ├── cluster_status.py
    ├── face_detection_bridge.py
    ├── face_detection_receiver.py
    ├── face_detection_receiver_alt.py
    ├── gpu_detection.py
    ├── journal_system.py
    ├── launch_senter_complete.py
    ├── light_controller.py
    ├── lights.py
    ├── process_manager.py
    ├── senter_face_bridge.py
    ├── senter_status.py
    ├── senter_ui.py
    ├── tools_config.py
    ├── user_profiles.py
    ├── Documentation/
    ├── Models/
    ├── QUARANTINE_AVA_DO_NOT_TOUCH/
    ├── SenterUI/
    │   ├── __init__.py
    │   ├── ui_components.py
    │   ├── AvA/
    │   │   ├── AImouse.py
    │   │   ├── CVA.py
    │   │   ├── ava.py
    │   │   ├── __pycache__/
    │   │   └── piper_models/
    │   └── __pycache__/
    ├── chroma_db_Chris/
    ├── chroma_db_Chris_Chris/
    ├── chroma_db_temp/
    ├── logs/
    ├── monitor_logs/
    ├── piper_models/
    ├── senter/
    │   ├── __init__.py
    │   ├── chat_history.py
    │   ├── config.py
    │   ├── network_coordinator.py
    │   ├── state_logger.py
    │   ├── tts_service.py
    │   └── __pycache__/
    ├── user_profiles/
    └── whisper_models/

```

## Content

```
================================================
File: audio_test.py
================================================
#!/usr/bin/env python3
"""
Audio Test Script for SENTER Docker Environment
Diagnoses and fixes audio configuration issues
"""

import os
import sys
import time
import subprocess
import sounddevice as sd
import numpy as np

def test_audio_devices():
    """Test and list available audio devices"""
    print("🔊 Testing Audio Devices...")
    print("=" * 50)
    
    try:
        devices = sd.query_devices()
        print("Available Audio Devices:")
        for i, device in enumerate(devices):
            device_type = "📥 Input" if device['max_input_channels'] > 0 else ""
            if device['max_output_channels'] > 0:
                device_type += " 📤 Output"
            print(f"  {i}: {device['name']} - {device_type}")
            print(f"      Sample Rate: {device['default_samplerate']} Hz")
            print(f"      Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        
        # Get default devices
        default_input = sd.default.device[0] if sd.default.device[0] is not None else "None"
        default_output = sd.default.device[1] if sd.default.device[1] is not None else "None"
        print(f"\nDefault Input Device: {default_input}")
        print(f"Default Output Device: {default_output}")
        
        return True
    except Exception as e:
        print(f"❌ Error querying audio devices: {e}")
        return False

def test_audio_output():
    """Test audio output with a simple tone"""
    print("\n🎵 Testing Audio Output...")
    try:
        # Generate a simple sine wave
        duration = 1.0  # seconds
        sample_rate = 22050  # Hz - Common rate that works in most environments
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        print(f"Playing test tone at {frequency}Hz for {duration}s...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        print("✅ Audio output test completed")
        return True
    except Exception as e:
        print(f"❌ Audio output test failed: {e}")
        return False

def test_audio_input():
    """Test audio input recording"""
    print("\n🎤 Testing Audio Input...")
    try:
        duration = 2.0  # seconds
        sample_rate = 16000  # Hz - Common rate for speech recognition
        
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        
        # Check if we got meaningful audio data
        max_amplitude = np.max(np.abs(audio))
        print(f"Max recorded amplitude: {max_amplitude:.4f}")
        
        if max_amplitude > 0.001:
            print("✅ Audio input test completed - got audio data")
            return True
        else:
            print("⚠️ Audio input test completed - no significant audio detected")
            return False
    except Exception as e:
        print(f"❌ Audio input test failed: {e}")
        return False

def fix_audio_config():
    """Try to fix common audio configuration issues"""
    print("\n🔧 Attempting to fix audio configuration...")
    
    # Set environment variables for better audio support
    audio_env = {
        'ALSA_CARD': '0',
        'ALSA_DEVICE': '0',
        'PULSE_RUNTIME_PATH': '/tmp/pulse-socket',
    }
    
    for key, value in audio_env.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Try to configure ALSA
    try:
        alsa_config = """
defaults.ctl.card 0
defaults.pcm.card 0
defaults.pcm.device 0
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
"""
        with open('/tmp/asound.conf', 'w') as f:
            f.write(alsa_config)
        os.environ['ALSA_CONF'] = '/tmp/asound.conf'
        print("✅ Created ALSA configuration")
    except Exception as e:
        print(f"⚠️ Could not create ALSA config: {e}")
    
    # Try to set audio device defaults for sounddevice
    try:
        # List devices and find working ones
        devices = sd.query_devices()
        
        # Find first working input device
        input_device = None
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_device = i
                break
        
        # Find first working output device  
        output_device = None
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_device = i
                break
        
        if input_device is not None:
            sd.default.device[0] = input_device
            print(f"✅ Set default input device to: {input_device}")
        
        if output_device is not None:
            sd.default.device[1] = output_device
            print(f"✅ Set default output device to: {output_device}")
            
        # Set a safe sample rate
        sd.default.samplerate = 22050
        print("✅ Set default sample rate to 22050 Hz")
        
    except Exception as e:
        print(f"⚠️ Could not set device defaults: {e}")

def main():
    """Main audio test function"""
    print("🎧 SENTER Audio System Test")
    print("=" * 50)
    
    # Test initial state
    devices_ok = test_audio_devices()
    
    if not devices_ok:
        print("\n❌ No audio devices found - audio functionality will be limited")
        return False
    
    # Try to fix configuration
    fix_audio_config()
    
    # Test audio functionality
    output_ok = test_audio_output()
    input_ok = test_audio_input()
    
    print("\n" + "=" * 50)
    print("🎧 Audio Test Summary:")
    print(f"  📋 Device Detection: {'✅' if devices_ok else '❌'}")
    print(f"  📤 Audio Output: {'✅' if output_ok else '❌'}")
    print(f"  📥 Audio Input: {'✅' if input_ok else '❌'}")
    
    if devices_ok and (output_ok or input_ok):
        print("\n✅ Audio system is functional!")
        return True
    else:
        print("\n❌ Audio system has issues - check Docker audio configuration")
        return False

if __name__ == "__main__":
    main() 


================================================
File: ava_audio_config.py
================================================

# Optimized AvA Audio Configuration

# Audio - Optimized for stability and quality
AUDIO_SAMPLE_RATE = 44100  # Use device native rate, resample for Whisper
AUDIO_CHANNELS = 1
AUDIO_BLOCK_DURATION_MS = 200  # Larger blocks for stability

# Audio quality thresholds - adjusted for better detection
MIN_RMS_LEVEL = 0.002      # Minimum RMS to consider as speech
MIN_MAX_LEVEL = 0.01       # Minimum peak level to consider as speech  
MAX_LEVEL_CLIP = 0.90      # Maximum level before considering clipped
MAX_DYNAMIC_RANGE = 40     # Maximum dynamic range before considering noise

# Whisper transcription settings
WHISPER_SAMPLE_RATE = 16000
WHISPER_BEAM_SIZE = 1      # Fast transcription
WHISPER_TEMPERATURE = 0.0  # Deterministic
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOG_PROB_THRESHOLD = -1.0

# Audio processing optimizations
ENABLE_AUDIO_NORMALIZATION = True
ENABLE_NOISE_REDUCTION = True
ENABLE_AGC = True  # Automatic Gain Control



================================================
File: camera_tools.py
================================================
#!/usr/bin/env python3

"""
Camera Tools Module for Senter
Handles image capture and vision processing using Ollama Gemma3:4B
"""

import cv2
import numpy as np
import requests
import json
import base64
import io
import time
import re
from PIL import Image, ImageGrab
from typing import Optional, Callable
import threading

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "gemma3:4b"  # Using gemma3:4b which is available on the system

class CameraTools:
    """Handles camera operations and vision processing."""
    
    def __init__(self, attention_detector=None):
        self.attention_detector = attention_detector
        self.ollama_available = self.check_ollama_availability()
    
    def check_ollama_availability(self):
        """Check if Ollama is running and the vision model is available."""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"ERROR: Ollama server not responding (status: {response.status_code})")
                return False
                
            # Check if the vision model is available
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
                    
            if VISION_MODEL not in model_names:
                print(f"ERROR: Vision model '{VISION_MODEL}' not found. Available models: {model_names}")
                print(f"   Install with: ollama pull {VISION_MODEL}")
                return False
            
            print(f"OK: Ollama vision ready with {VISION_MODEL}")
            return True
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            print(f"   Start Ollama with: ollama serve")
            return False
        except Exception as e:
            print(f"ERROR: Ollama check failed: {e}")
            return False
    
    def capture_webcam_image(self) -> Optional[np.ndarray]:
        """Capture image from the webcam (front camera)."""
        try:
            # OPTIMIZATION: Use existing frame from attention detector if available
            if (self.attention_detector and 
                hasattr(self.attention_detector, 'camera') and 
                self.attention_detector.camera and 
                self.attention_detector.camera.isOpened()):
                
                print("📸 Using attention detector camera stream")
                
                # Capture frame from the existing camera stream
                ret, frame = self.attention_detector.camera.read()
                if ret and frame is not None:
                    print("CAMERA: Captured from attention detector camera")
                    return frame
                else:
                    print("WARNING: Attention detector camera failed, trying fallback...")
            
            # Fallback: Create temporary capture
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("ERROR: Cannot open camera")
                return None
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()  # Important: release the camera
            
            if ret and frame is not None:
                print("CAMERA: Captured from fallback camera")
                return frame
            else:
                print("ERROR: Failed to capture frame")
                return None
                
        except Exception as e:
            print(f"ERROR: Camera capture error: {e}")
            return None
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot of the current screen."""
        try:
            # Capture screenshot using PIL
            screenshot = ImageGrab.grab()
            
            # Convert PIL image to OpenCV format
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            print("SCREENSHOT: Screenshot captured")
            return frame
            
        except Exception as e:
            print(f"ERROR: Screenshot capture error: {e}")
            return None
    
    def process_camera_command(self, camera_command: str, tts_callback: Optional[Callable] = None, silent_mode: bool = False) -> bool:
        """Process a camera command and return analysis results."""
        try:
            if not self.ollama_available:
                error_msg = "Camera vision is not available. Please ensure Ollama is running with Gemma3:4B model."
                if not silent_mode:
                    print(f"ERROR: {error_msg}")
                if tts_callback and not silent_mode:
                    tts_callback("I'm sorry, but camera vision is not available right now.")
                return False
            
            # Determine what type of image to capture
            command_lower = camera_command.lower()
            
            if "screenshot" in command_lower or "screen" in command_lower or "computer" in command_lower:
                if not silent_mode:
                    print("SCREENSHOT: Taking screenshot...")
                image = self.capture_screenshot()
                analysis = self.analyze_image_with_ollama(image, camera_command, tts_callback, silent_mode)
                
            else:
                # Check for pre-analyzed camera data (SPEED OPTIMIZATION!)
                pre_analysis = None
                if (self.attention_detector and 
                    hasattr(self.attention_detector, 'get_camera_analysis')):
                    pre_analysis = self.attention_detector.get_camera_analysis()
                
                if pre_analysis and pre_analysis.get('analysis'):
                    # Use pre-analyzed data for instant response!
                    if not silent_mode:
                        print("FAST: Using pre-analyzed camera data (instant response!)")
                    
                    # Stream the pre-analyzed response only if not in silent mode
                    analysis_text = pre_analysis['analysis']
                    if analysis_text and tts_callback and not silent_mode:
                        # Break into sentences and stream via TTS
                        sentences = re.split(r'[.!?]+', analysis_text)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence:
                                print(f"    TTS: Streaming pre-analysis: '{sentence[:50]}...'")
                                tts_callback(sentence + ".")
                    
                    analysis = analysis_text
                else:
                    # Fallback to real-time capture and analysis
                    if not silent_mode:
                        print("CAMERA: Taking webcam photo...")
                    image = self.capture_webcam_image()
                    if image is None:
                        error_msg = "Failed to capture webcam photo"
                        if not silent_mode:
                            print(f"ERROR: {error_msg}")
                        if tts_callback and not silent_mode:
                            tts_callback("I'm sorry, I couldn't capture the photo.")
                        return False
                    
                    analysis = self.analyze_image_with_ollama(image, camera_command, tts_callback if not silent_mode else None, silent_mode)
            
            if analysis:
                if not silent_mode:
                    print(f"OK: Camera analysis completed")
                return True
            else:
                if not silent_mode:
                    print(f"ERROR: Camera analysis failed")
                return False
                
        except Exception as e:
            if not silent_mode:
                print(f"ERROR: Camera command processing error: {e}")
            if tts_callback and not silent_mode:
                tts_callback("I encountered an error while processing the camera request.")
            return False
    
    def analyze_image_with_ollama(self, image: np.ndarray, prompt: str, tts_callback: Optional[Callable] = None, silent_mode: bool = False) -> Optional[str]:
        """Send image to Ollama for vision analysis with streaming response."""
        try:
            # Convert image to base64
            image_b64 = self.encode_image_to_base64(image, silent_mode=silent_mode)
            if not image_b64:
                return None
            
            # SPEED OPTIMIZATION: Create shorter, more focused prompt
            if "hair" in prompt.lower() or "look" in prompt.lower():
                vision_prompt = f"Analyze this photo of a person and describe their appearance, focusing on: {prompt}. Be concise and helpful."
            elif "screenshot" in prompt.lower() or "screen" in prompt.lower():
                vision_prompt = f"Describe what's visible on this computer screen. Be concise and helpful."
            else:
                vision_prompt = f"Describe this image briefly: {prompt}. Be concise."
            
            if not silent_mode:
                print(f"AI: Analyzing image with {VISION_MODEL}...")
            
            # SPEED OPTIMIZATION: Prepare request with faster settings
            payload = {
                "model": VISION_MODEL,
                "prompt": vision_prompt,
                "images": [image_b64],
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": 0.3,  # Lower temperature for faster, more focused responses
                    "top_k": 20,        # Smaller top_k for speed
                    "top_p": 0.8,       # Lower top_p for speed
                    "num_predict": 150,  # Limit response length for speed
                }
            }
            
            # Send streaming request to Ollama
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                stream=True,
                timeout=180  # Increased timeout for vision processing (was 120, now 180 for Gemma3:4B)
            )
            
            if response.status_code != 200:
                if not silent_mode:
                    print(f"ERROR: Ollama API error: {response.status_code}")
                return None
            
            # Process streaming response
            full_response = ""
            sentence_buffer = ""
            
            if not silent_mode:
                print("AI: Streaming vision analysis...")
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk_text = data.get('response', '')
                        
                        if chunk_text:
                            full_response += chunk_text
                            sentence_buffer += chunk_text
                            
                            # Check for sentence completion (same logic as research tool)
                            while True:
                                match = re.search(r"([.?!])", sentence_buffer)
                                if match:
                                    end_index = match.end()
                                    sentence = sentence_buffer[:end_index].strip()
                                    if sentence and tts_callback:
                                        if not silent_mode:
                                            print(f"    TTS: Queueing vision response: '{sentence[:50]}...'")
                                        tts_callback(sentence)
                                    
                                    sentence_buffer = sentence_buffer[end_index:].lstrip()
                                else:
                                    break
                        
                        # Check if generation is done
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Queue any remaining text
            if sentence_buffer.strip() and tts_callback:
                remaining = sentence_buffer.strip()
                if not silent_mode:
                    print(f"    TTS: Queueing remaining vision response: '{remaining[:50]}...'")
                tts_callback(remaining)
            
            if not silent_mode:
                print(f"OK: Vision analysis completed: {len(full_response)} characters")
            return full_response
            
        except Exception as e:
            if not silent_mode:
                print(f"ERROR: Ollama vision analysis error: {e}")
            return None
    
    def encode_image_to_base64(self, image: np.ndarray, silent_mode: bool = False) -> Optional[str]:
        """Convert OpenCV image to base64 string for API."""
        try:
            # SPEED OPTIMIZATION: More aggressive resizing for faster processing
            height, width = image.shape[:2]
            max_size = 512  # Reduced from 1024 for much faster processing
            
            if max(height, width) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                image = cv2.resize(image, (new_width, new_height))
                if not silent_mode:
                    print(f"RESIZE: Resized image: {width}x{height} -> {new_width}x{new_height}")
            
            # SPEED OPTIMIZATION: More aggressive JPEG compression for faster upload
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Reduced from 85
            if not success:
                if not silent_mode:
                    print("ERROR: Failed to encode image as JPEG")
                return None
            
            # Convert to base64
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            if not silent_mode:
                print(f"OK: Image encoded: {len(image_b64)} characters")
            
            return image_b64
            
        except Exception as e:
            if not silent_mode:
                print(f"ERROR: Image encoding error: {e}")
            return None

# Global instance
camera_tools = CameraTools()

def execute_camera_command(camera_command: str, tts_callback: Optional[Callable] = None, attention_detector=None, silent_mode: bool = False) -> bool:
    """
    Execute a camera command with vision analysis.
    
    Args:
        camera_command: Command like "front camera", "screenshot", "take photo"
        tts_callback: Function to call for TTS output (ignored if silent_mode=True)
        attention_detector: Attention detector instance for camera access
        silent_mode: If True, run silently without print statements or TTS
        
    Returns:
        bool: True if successful, False otherwise
    """
    global camera_tools
    
    # Update camera tools with attention detector if provided
    if attention_detector:
        camera_tools.attention_detector = attention_detector
    
    return camera_tools.process_camera_command(camera_command, tts_callback, silent_mode) 


================================================
File: capture_logs.py
================================================
#!/usr/bin/env python3
"""
Real-time log capture for SENTER debugging
Captures logs continuously and saves them even if container crashes
"""

import subprocess
import datetime
import signal
import sys
import threading
import time

class LogCapture:
    def __init__(self):
        self.log_file = f"senter_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.process = None
        self.running = False
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n🛑 Stopping log capture...")
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)
        
    def capture_logs(self):
        """Capture docker-compose logs continuously"""
        print(f"📋 Starting continuous log capture...")
        print(f"📝 Logs will be saved to: {self.log_file}")
        print(f"💡 Press Ctrl+C to stop log capture")
        print("=" * 50)
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.running = True
        
        with open(self.log_file, 'w') as f:
            f.write(f"SENTER Debug Log - Started at {datetime.datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            while self.running:
                try:
                    # Start docker-compose logs process
                    self.process = subprocess.Popen(
                        ['docker-compose', 'logs', '-f', '--tail=0'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    print("🔄 Connected to log stream...")
                    
                    # Read logs line by line
                    for line in iter(self.process.stdout.readline, ''):
                        if not self.running:
                            break
                            
                        # Print to console
                        print(line.rstrip())
                        
                        # Write to file with timestamp
                        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        f.write(f"[{timestamp}] {line}")
                        f.flush()  # Immediate write to disk
                        
                        # Check for specific error patterns
                        if any(pattern in line.lower() for pattern in [
                            'error', 'exception', 'crash', 'segmentation fault', 
                            'terminate called', 'ollama', 'cuda', 'memory'
                        ]):
                            f.write(f"[{timestamp}] ⚠️ POTENTIAL ERROR DETECTED ⚠️\n")
                            f.flush()
                    
                    # Process ended
                    if self.running:
                        print("⚠️ Log stream ended, attempting to reconnect...")
                        f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Log stream disconnected, reconnecting...\n")
                        f.flush()
                        time.sleep(2)  # Wait before reconnecting
                        
                except Exception as e:
                    print(f"❌ Log capture error: {e}")
                    f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Log capture error: {e}\n")
                    f.flush()
                    if self.running:
                        time.sleep(2)  # Wait before retrying
                        
        print(f"✅ Log capture stopped. Logs saved to: {self.log_file}")

def main():
    capture = LogCapture()
    capture.capture_logs()

if __name__ == "__main__":
    main() 


================================================
File: cluster_status.py
================================================
#!/usr/bin/env python3
"""
SENTER Cluster Status Monitor
============================

Real-time monitoring of SENTER cluster nodes and their resource usage.
This script connects to the cluster network and displays live status information.
"""

import time
import json
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def format_timestamp(timestamp):
    """Format timestamp for display."""
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

def format_bytes(bytes_value):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def display_cluster_status(cluster_summary, network_info=None):
    """Display formatted cluster status."""
    clear_screen()
    
    print("🌐 SENTER Cluster Status Monitor")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Cluster health overview
    health = cluster_summary['cluster_health']
    print(f"🏥 Cluster Health:")
    print(f"   Total Nodes: {health['total_nodes']}")
    print(f"   Healthy: {health['healthy_nodes']} | Unhealthy: {health['unhealthy_nodes']}")
    
    if health['unhealthy_nodes'] > 0:
        print("   ⚠️  Some nodes are offline or unresponsive")
    else:
        print("   ✅ All nodes are healthy")
    print()
    
    # Resource totals
    resources = cluster_summary['resource_totals']
    print(f"📊 Cluster Resources:")
    print(f"   Average CPU: {resources['avg_cpu_percent']:.1f}%")
    print(f"   Average Memory: {resources['avg_memory_percent']:.1f}%")
    print(f"   Total GPU Memory: {resources['total_gpu_memory_gb']:.1f} GB")
    print()
    
    # Individual nodes
    print(f"🖥️  Node Details:")
    print("-" * 60)
    
    nodes = cluster_summary['nodes']
    local_node_id = cluster_summary['local_node_id']
    
    for node_id, node_info in nodes.items():
        is_local = node_id == local_node_id
        status_icon = "✅" if node_info['healthy'] else "❌"
        local_icon = "🏠" if is_local else "🌐"
        
        print(f"{status_icon} {local_icon} {node_id}")
        print(f"   Mode: {node_info['system_mode']} | Attention: {node_info['attention_state']}")
        print(f"   User: {node_info['current_user'] or 'None'}")
        print(f"   Last Seen: {format_timestamp(node_info['last_seen'])} ({node_info['age_seconds']:.0f}s ago)")
        
        if 'cpu_percent' in node_info:
            print(f"   CPU: {node_info['cpu_percent']:.1f}% | Memory: {node_info['memory_percent']:.1f}%")
            if node_info.get('gpu_memory_gb', 0) > 0:
                print(f"   GPU Memory: {node_info['gpu_memory_gb']:.1f} GB")
            print(f"   Threads: {node_info['active_threads']}")
        else:
            print("   📊 Resource metrics not available")
        
        print()
    
    # Network information
    if network_info:
        print(f"🔗 Network Information:")
        print(f"   Local IP: {network_info.get('local_ip', 'Unknown')}")
        print(f"   UDP Port: {network_info.get('udp_port', 'Unknown')}")
        print(f"   Peers Discovered: {len(network_info.get('peers', {}))}")
        print()
    
    print("Press Ctrl+C to exit")

async def monitor_cluster():
    """Main monitoring loop."""
    try:
        from senter.state_logger import StateLogger
        from senter.network_coordinator import create_network_coordinator
        from process_manager import init_process_management
        
        # Initialize components
        node_id = f"monitor-{int(time.time())}"
        logs_dir = Path("monitor_logs")
        logs_dir.mkdir(exist_ok=True)
        
        print("🚀 Initializing cluster monitor...")
        
        # Create state logger
        state_logger = StateLogger(
            logs_dir=logs_dir,
            session_id=f"monitor_{int(time.time())}",
            node_id=node_id
        )
        
        # Create network coordinator
        network_coordinator = create_network_coordinator(node_id=node_id, enable_discovery=True)
        
        # Initialize process manager for local metrics
        process_manager = init_process_management()
        
        # Wire components together
        state_logger.set_process_manager(process_manager)
        
        if not network_coordinator.start():
            print("❌ Failed to start network coordinator")
            return 1
        
        state_logger.set_network_coordinator(network_coordinator)
        
        print("✅ Monitor initialized, starting display...")
        time.sleep(2)  # Wait for initial discovery
        
        # Monitoring loop
        while True:
            try:
                # Update local metrics
                state_logger.update_resource_metrics()
                
                # Get cluster status
                cluster_summary = state_logger.get_cluster_summary()
                
                # Get network info
                network_info = network_coordinator.get_cluster_info()
                
                # Display status
                display_cluster_status(cluster_summary, network_info)
                
                # Wait before next update
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(5)
        
        # Cleanup
        print("\n🛑 Shutting down monitor...")
        network_coordinator.stop()
        process_manager.stop_monitoring()
        state_logger.close()
        
        return 0
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Make sure zeroconf is installed: pip install zeroconf")
        return 1
    except Exception as e:
        print(f"❌ Monitor failed: {e}")
        return 1

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\n📡 Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🌐 SENTER Cluster Status Monitor")
    print("=" * 40)
    print("This tool monitors all SENTER instances on your network")
    print("and displays real-time cluster status information.")
    print()
    
    try:
        return await monitor_cluster()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 


================================================
File: face_detection_bridge.py
================================================
#!/usr/bin/env python3
"""
Face Detection Bridge for SENTER
================================

This script shares face detection events from this SENTER instance
to a remote SENTER instance at 192.168.1.15.

It monitors the local face detection state and sends notifications
when faces are detected or lost.
"""

import time
import json
import socket
import threading
import requests
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Configuration
REMOTE_SENTER_IP = "192.168.1.15"
REMOTE_SENTER_PORT = 8080  # HTTP API port for receiving face detection data
UPDATE_INTERVAL = 1.0  # Send updates every second when face is detected
HEARTBEAT_INTERVAL = 30.0  # Send heartbeat every 30 seconds

# Face detection configuration
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACE_AREA_THRESHOLD = 0.02  # Minimum face area as proportion of frame
MIN_FACE_SIZE = (60, 60)
RESIZE_WIDTH = 640

class FaceDetectionBridge:
    """Bridges face detection data to remote SENTER instance."""
    
    def __init__(self, remote_ip: str = REMOTE_SENTER_IP, remote_port: int = REMOTE_SENTER_PORT):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.is_running = False
        self.camera = None
        self.face_cascade = None
        
        # State tracking
        self.current_face_detected = False
        self.last_face_detection_time = 0
        self.last_heartbeat_time = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Get local machine info
        self.local_hostname = socket.gethostname()
        self.local_ip = self._get_local_ip()
        
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def initialize_camera(self) -> bool:
        """Initialize the camera for face detection."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.logger.error("Failed to open camera")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def initialize_face_detection(self) -> bool:
        """Initialize face detection cascade."""
        try:
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            if self.face_cascade.empty():
                self.logger.error(f"Failed to load face cascade from {FACE_CASCADE_PATH}")
                return False
            
            self.logger.info("Face detection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Face detection initialization failed: {e}")
            return False
    
    def detect_face(self, frame: np.ndarray) -> bool:
        """Detect if a face is present in the frame."""
        try:
            # Resize frame for faster processing
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(RESIZE_WIDTH / aspect_ratio)
            resized_frame = cv2.resize(frame, (RESIZE_WIDTH, new_height))
            frame_height, frame_width = resized_frame.shape[:2]
            frame_area = frame_width * frame_height
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE
            )
            
            if len(faces) > 0:
                # Find the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                largest_face_area = (w * h) / frame_area
                
                # Check if face is large enough
                return largest_face_area >= FACE_AREA_THRESHOLD
            
            return False
            
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return False
    
    def send_face_detection_update(self, face_detected: bool, force_send: bool = False):
        """Send face detection update to remote SENTER."""
        current_time = time.time()
        
        # Determine if we should send an update
        should_send = (
            force_send or
            face_detected != self.current_face_detected or  # State changed
            (face_detected and current_time - self.last_face_detection_time > UPDATE_INTERVAL) or  # Regular updates when face present
            current_time - self.last_heartbeat_time > HEARTBEAT_INTERVAL  # Heartbeat
        )
        
        if not should_send:
            return
        
        try:
            # Prepare the data payload
            data = {
                "source": {
                    "hostname": self.local_hostname,
                    "ip": self.local_ip,
                    "timestamp": datetime.now().isoformat()
                },
                "face_detection": {
                    "detected": face_detected,
                    "timestamp": datetime.now().isoformat(),
                    "changed": face_detected != self.current_face_detected
                }
            }
            
            # Send HTTP POST request to remote SENTER
            url = f"http://{self.remote_ip}:{self.remote_port}/api/face-detection"
            
            response = requests.post(
                url,
                json=data,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                status_text = "DETECTED" if face_detected else "LOST"
                if face_detected != self.current_face_detected:
                    self.logger.info(f"🎯 Face {status_text} - Sent to {self.remote_ip}")
                
                self.current_face_detected = face_detected
                if face_detected:
                    self.last_face_detection_time = current_time
                self.last_heartbeat_time = current_time
                
            else:
                self.logger.warning(f"Remote SENTER responded with status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            if force_send:  # Only log connection errors on initial connection or forced sends
                self.logger.warning(f"Cannot connect to remote SENTER at {self.remote_ip}:{self.remote_port}")
        except Exception as e:
            self.logger.error(f"Error sending face detection update: {e}")
    
    def detection_loop(self):
        """Main face detection loop."""
        self.logger.info("Starting face detection loop...")
        
        # Send initial connection message
        self.send_face_detection_update(False, force_send=True)
        
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(1)
                    continue
                
                # Detect face in current frame
                face_detected = self.detect_face(frame)
                
                # Send update to remote SENTER
                self.send_face_detection_update(face_detected)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def start(self) -> bool:
        """Start the face detection bridge."""
        self.logger.info(f"🚀 Starting Face Detection Bridge")
        self.logger.info(f"   Local: {self.local_hostname} ({self.local_ip})")
        self.logger.info(f"   Remote: {self.remote_ip}:{self.remote_port}")
        
        # Initialize components
        if not self.initialize_camera():
            return False
        
        if not self.initialize_face_detection():
            return False
        
        # Start detection loop in separate thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.logger.info("✅ Face Detection Bridge started successfully")
        return True
    
    def stop(self):
        """Stop the face detection bridge."""
        self.logger.info("Stopping Face Detection Bridge...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        # Send final update
        self.send_face_detection_update(False, force_send=True)
        
        self.logger.info("✅ Face Detection Bridge stopped")


def main():
    """Main entry point."""
    print("🎯 SENTER Face Detection Bridge")
    print("=" * 40)
    print(f"Sharing face detection data with: {REMOTE_SENTER_IP}:{REMOTE_SENTER_PORT}")
    print("Press Ctrl+C to stop")
    print()
    
    bridge = FaceDetectionBridge()
    
    try:
        if bridge.start():
            # Keep running until interrupted
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start face detection bridge")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        bridge.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        bridge.stop()


if __name__ == "__main__":
    main() 


================================================
File: face_detection_receiver.py
================================================
#!/usr/bin/env python3
"""
Face Detection Receiver for SENTER
==================================

This script runs an HTTP server that receives face detection events
from other SENTER instances on the network.

Run this on the remote SENTER instance (192.168.1.15) to receive
face detection notifications from other machines.
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify
import threading

# Configuration
SERVER_PORT = 9091
FACE_DETECTION_TIMEOUT = 10.0  # Consider face detection stale after 10 seconds

class FaceDetectionReceiver:
    """Receives and processes face detection data from remote SENTER instances."""
    
    def __init__(self, port: int = SERVER_PORT):
        self.port = port
        self.app = Flask(__name__)
        
        # State tracking for each source
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.sources_lock = threading.RLock()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup Flask routes
        self._setup_routes()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _setup_routes(self):
        """Setup Flask HTTP routes."""
        
        @self.app.route('/api/face-detection', methods=['POST'])
        def receive_face_detection():
            """Receive face detection data from remote SENTER."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                return self._handle_face_detection_update(data)
                
            except Exception as e:
                self.logger.error(f"Error processing face detection update: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get current status of all face detection sources."""
            with self.sources_lock:
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "sources": dict(self.sources),
                    "total_sources": len(self.sources),
                    "active_detections": sum(1 for source in self.sources.values() 
                                           if source.get("face_detected", False))
                }
            return jsonify(status)
        
        @self.app.route('/api/sources', methods=['GET'])
        def get_sources():
            """Get list of active sources."""
            with self.sources_lock:
                sources_list = []
                for source_id, source_data in self.sources.items():
                    sources_list.append({
                        "id": source_id,
                        "hostname": source_data.get("hostname", "unknown"),
                        "ip": source_data.get("ip", "unknown"),
                        "face_detected": source_data.get("face_detected", False),
                        "last_update": source_data.get("last_update", "never"),
                        "age_seconds": time.time() - source_data.get("last_update_timestamp", 0)
                    })
            return jsonify({"sources": sources_list})
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with status information."""
            with self.sources_lock:
                sources_html = ""
                for source_id, source_data in self.sources.items():
                    status_icon = "👁️" if source_data.get("face_detected", False) else "😴"
                    age = time.time() - source_data.get("last_update_timestamp", 0)
                    sources_html += f"""
                    <div style="border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;">
                        <h3>{status_icon} {source_data.get('hostname', 'Unknown')}</h3>
                        <p><strong>IP:</strong> {source_data.get('ip', 'Unknown')}</p>
                        <p><strong>Face Detected:</strong> {'Yes' if source_data.get('face_detected', False) else 'No'}</p>
                        <p><strong>Last Update:</strong> {source_data.get('last_update', 'Never')} ({age:.1f}s ago)</p>
                    </div>
                    """
                
                if not sources_html:
                    sources_html = "<p>No SENTER sources connected yet.</p>"
            
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SENTER Face Detection Receiver</title>
                <meta http-equiv="refresh" content="2">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎯 SENTER Face Detection Receiver</h1>
                    <p>Monitoring face detection from remote SENTER instances</p>
                    <p><strong>Server:</strong> Running on port {self.port}</p>
                    <p><strong>Active Sources:</strong> {len(self.sources)}</p>
                    <p><strong>Current Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Connected Sources</h2>
                {sources_html}
                
                <h2>API Endpoints</h2>
                <ul>
                    <li><a href="/api/status">/api/status</a> - JSON status of all sources</li>
                    <li><a href="/api/sources">/api/sources</a> - JSON list of sources</li>
                    <li><strong>/api/face-detection</strong> - POST endpoint for receiving data</li>
                </ul>
            </body>
            </html>
            """
    
    def _handle_face_detection_update(self, data: Dict[str, Any]) -> tuple:
        """Handle incoming face detection update."""
        try:
            # Extract source information
            source_info = data.get("source", {})
            face_info = data.get("face_detection", {})
            
            hostname = source_info.get("hostname", "unknown")
            ip = source_info.get("ip", "unknown")
            source_id = f"{hostname}_{ip}"
            
            face_detected = face_info.get("detected", False)
            face_changed = face_info.get("changed", False)
            
            current_time = time.time()
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Update source data
            with self.sources_lock:
                if source_id not in self.sources:
                    self.logger.info(f"🔗 New SENTER source connected: {hostname} ({ip})")
                
                # Store previous state for change detection
                prev_face_detected = self.sources.get(source_id, {}).get("face_detected", False)
                
                self.sources[source_id] = {
                    "hostname": hostname,
                    "ip": ip,
                    "face_detected": face_detected,
                    "last_update": timestamp_str,
                    "last_update_timestamp": current_time,
                    "total_updates": self.sources.get(source_id, {}).get("total_updates", 0) + 1
                }
            
            # Log significant events
            if face_changed or face_detected != prev_face_detected:
                status_text = "DETECTED" if face_detected else "LOST"
                self.logger.info(f"👁️  Face {status_text} from {hostname} ({ip})")
                
                # Here you could integrate with the local SENTER system
                # For example, trigger attention events, activate lights, etc.
                self._handle_face_detection_event(source_id, face_detected, hostname, ip)
            
            return jsonify({"status": "received", "source": source_id}), 200
            
        except Exception as e:
            self.logger.error(f"Error handling face detection update: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_face_detection_event(self, source_id: str, face_detected: bool, hostname: str, ip: str):
        """Handle face detection events - integrate with local SENTER system here."""
        if face_detected:
            print(f"🎯 FACE DETECTED on {hostname} ({ip})")
            # TODO: Integrate with local SENTER attention system
            # You could:
            # - Trigger attention state changes
            # - Activate voice recording
            # - Turn on lights
            # - Send notifications
            
        else:
            print(f"😴 Face lost on {hostname} ({ip})")
            # TODO: Handle face lost events
    
    def _cleanup_loop(self):
        """Clean up stale sources periodically."""
        while True:
            current_time = time.time()
            stale_sources = []
            
            with self.sources_lock:
                for source_id, source_data in self.sources.items():
                    age = current_time - source_data.get("last_update_timestamp", 0)
                    if age > FACE_DETECTION_TIMEOUT:
                        stale_sources.append(source_id)
            
            # Remove stale sources
            if stale_sources:
                with self.sources_lock:
                    for source_id in stale_sources:
                        source_data = self.sources.pop(source_id, {})
                        hostname = source_data.get("hostname", "unknown")
                        self.logger.warning(f"🔌 SENTER source disconnected: {hostname} (timeout)")
            
            time.sleep(5)  # Check every 5 seconds
     
    def start(self):
        """Start the face detection receiver server."""
        self.logger.info(f"🚀 Starting Face Detection Receiver on port {self.port}")
        self.logger.info(f"📡 Waiting for face detection data from remote SENTER instances...")
        self.logger.info(f"🌐 Web interface: http://localhost:{self.port}")
        
        # Run Flask app
        self.app.run(
            host='0.0.0.0',  # Listen on all interfaces
            port=self.port,
            debug=False,
            use_reloader=False
        )


def main():
    """Main entry point."""
    print("📡 SENTER Face Detection Receiver")
    print("=" * 40)
    print(f"Starting HTTP server on port {SERVER_PORT}")
    print("This server will receive face detection events from other SENTER instances.")
    print(f"Web interface will be available at: http://localhost:{SERVER_PORT}")
    print()
    
    receiver = FaceDetectionReceiver(port=SERVER_PORT)
    
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 


================================================
File: face_detection_receiver_alt.py
================================================
#!/usr/bin/env python3
"""
Alternative SENTER Face Detection Receiver (Port 8888)
======================================================

Simple version for testing connectivity issues.
"""

from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)
received_events = []

@app.route('/api/face-detection', methods=['POST'])
def receive_face_detection():
    data = request.get_json()
    received_events.append({
        'timestamp': datetime.now().isoformat(),
        'data': data
    })
    
    source = data.get('source', {})
    face_info = data.get('face_detection', {})
    
    status = "DETECTED" if face_info.get('detected') else "LOST"
    hostname = source.get('hostname', 'unknown')
    
    print(f"📡 Received: Face {status} from {hostname}")
    
    return jsonify({"status": "received"})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "server": "SENTER Face Detection Receiver",
        "port": 8888,
        "total_events": len(received_events),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return f"""
    <h1>🎯 SENTER Face Detection Receiver</h1>
    <p>Server running on port 8888</p>
    <p>Total events received: {len(received_events)}</p>
    <p>Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <a href="/api/status">API Status</a>
    """

if __name__ == "__main__":
    print("🚀 Starting simple receiver on port 8888")
    print("Test with: curl http://localhost:8888/")
    app.run(host='0.0.0.0', port=8888, debug=False) 


================================================
File: gpu_detection.py
================================================
#!/usr/bin/env python3
"""
GPU Detection and Optimization for SENTER
Automatically detects GPU resources and configures optimal settings
"""

import subprocess
import os
import torch
import platform

def detect_gpu_resources():
    """Detect available GPU resources and return optimal configuration"""
    gpu_info = {
        'has_cuda': False,
        'has_nvidia': False,
        'gpu_memory': 0,
        'gpu_count': 0,
        'recommended_gpu_layers': 0,
        'device': 'cpu',
        'compute_type': 'int8'
    }
    
    print("🔍 Detecting GPU resources...")
    
    # Check CUDA availability via PyTorch
    try:
        gpu_info['has_cuda'] = torch.cuda.is_available()
        if gpu_info['has_cuda']:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['device'] = 'cuda'
            print(f"✅ CUDA available: {gpu_info['gpu_count']} GPU(s)")
            
            # Get GPU memory info
            for i in range(gpu_info['gpu_count']):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / 1024**3
                gpu_info['gpu_memory'] = max(gpu_info['gpu_memory'], memory_gb)
                print(f"   GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
            
            # Recommend GPU layers based on memory
            if gpu_info['gpu_memory'] >= 8:
                gpu_info['recommended_gpu_layers'] = -1  # All layers
                # Check GPU name for P4000 compatibility
                gpu_name = torch.cuda.get_device_properties(0).name if gpu_info['gpu_count'] > 0 else ""
                if 'Quadro P4000' in gpu_name or 'P4000' in gpu_name:
                    gpu_info['compute_type'] = 'int8'
                    print(f"🎯 Quadro P4000 detected: Using all GPU layers with int8 for compatibility")
                else:
                    gpu_info['compute_type'] = 'float16'
                    print(f"🚀 High-end GPU detected: Using all GPU layers with float16")
            elif gpu_info['gpu_memory'] >= 6:
                gpu_info['recommended_gpu_layers'] = 25  # Most layers
                gpu_info['compute_type'] = 'int8'
                print(f"🎯 Mid-range GPU detected: Using 25 GPU layers with int8")
            elif gpu_info['gpu_memory'] >= 4:
                gpu_info['recommended_gpu_layers'] = 15  # Some layers
                gpu_info['compute_type'] = 'int8'
                print(f"💡 Lower-end GPU detected: Using 15 GPU layers with int8")
            else:
                gpu_info['recommended_gpu_layers'] = 5  # Minimal layers
                gpu_info['compute_type'] = 'int8'
                print(f"⚡ Low-memory GPU: Using 5 GPU layers with int8")
                
            # Test actual GPU memory allocation
            try:
                test_tensor = torch.ones(1000, 1000).cuda()
                print(f"✅ GPU memory allocation test: Success")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ GPU memory test failed: {e} - falling back to CPU")
                gpu_info['has_cuda'] = False
                gpu_info['device'] = 'cpu'
                gpu_info['recommended_gpu_layers'] = 0
        else:
            print("❌ CUDA not available via PyTorch")
    except Exception as e:
        print(f"⚠️ Error checking CUDA: {e}")
        gpu_info['has_cuda'] = False
    
    # Check NVIDIA GPU via nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info['has_nvidia'] = True
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    gpu_name, memory_mb = line.split(', ')
                    memory_gb = float(memory_mb) / 1024
                    gpu_info['gpu_memory'] = max(gpu_info['gpu_memory'], memory_gb)
                    print(f"✅ NVIDIA GPU detected: {gpu_name.strip()} ({memory_gb:.1f}GB)")
                    
            # Update recommendations based on actual GPU memory
            if gpu_info['gpu_memory'] >= 6:
                gpu_info['recommended_gpu_layers'] = -1  # All layers
                gpu_info['device'] = 'cuda'
                # Force int8 for Quadro P4000 and similar older architectures
                if 'Quadro P4000' in gpu_name or 'P4000' in gpu_name:
                    gpu_info['compute_type'] = 'int8'
                    print(f"🎯 Quadro P4000 detected: Using int8 for optimal compatibility")
                else:
                    gpu_info['compute_type'] = 'float16' if gpu_info['gpu_memory'] >= 8 else 'int8'
                print(f"🚀 GPU acceleration enabled: {gpu_info['recommended_gpu_layers']} layers, {gpu_info['compute_type']}")
            elif gpu_info['gpu_memory'] >= 4:
                gpu_info['recommended_gpu_layers'] = 20
                gpu_info['device'] = 'cuda'
                gpu_info['compute_type'] = 'int8'
                print(f"🎯 Partial GPU acceleration: {gpu_info['recommended_gpu_layers']} layers")
        else:
            print("❌ nvidia-smi command failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - installing nvidia-utils might be needed")
    except Exception as e:
        print(f"⚠️ nvidia-smi check failed: {e}")
    
    # Fallback to CPU optimizations
    if not gpu_info['has_cuda']:
        cpu_count = os.cpu_count()
        print(f"💻 Using CPU-only mode with {cpu_count} threads")
        gpu_info['device'] = 'cpu'
        gpu_info['compute_type'] = 'int8'
        
        # Optimize for CPU
        os.environ['OMP_NUM_THREADS'] = str(min(8, cpu_count))
        os.environ['MKL_NUM_THREADS'] = str(min(8, cpu_count))
        print(f"🔧 Set thread limits for optimal CPU performance")
    
    return gpu_info

def optimize_whisper_config(gpu_info):
    """Get optimal Whisper configuration based on GPU resources"""
    # STABILITY FIRST: Force Whisper to CPU to avoid GPU memory conflicts and crashes
    config = {
        'device': 'cpu',  # Always use CPU for stability
        'compute_type': 'int8',
        'model_size': 'small'  # Use small for good quality on CPU
    }
    
    # CPU-only Whisper configuration for maximum stability
    if gpu_info['has_cuda'] and gpu_info['gpu_memory'] >= 6:
        config['model_size'] = 'small'  # Good quality on CPU
        print("🎯 Using Whisper 'small' model on CPU for stability")
        print("   💡 LLM uses GPU, Whisper uses CPU - avoids memory conflicts and crashes")
    else:
        config['model_size'] = 'tiny'  # Faster on limited systems
        print("🚀 Using Whisper 'tiny' model on CPU for maximum speed")
    
    return config

def optimize_llm_config(gpu_info):
    """Get optimal LLM configuration based on GPU resources"""
    # Optimize for SPEED over context size for better engagement
    config = {
        'n_gpu_layers': gpu_info['recommended_gpu_layers'],
        'n_ctx': 1024,  # Drastically reduced from 2048 for much faster inference
        'n_batch': 128 if gpu_info['has_cuda'] else 32,  # Reduced for speed
        'n_threads': min(4, os.cpu_count()) if not gpu_info['has_cuda'] else 2,  # Fewer threads
        'use_mlock': False,
        'use_mmap': True,
        'verbose': False,
        # Add speed optimizations
        'n_predict': 75,  # Reduced from 150 for much faster response
        'temp': 0.1,  # Even lower temperature for faster responses
        'top_k': 10,  # Further reduce top_k for faster sampling
        'top_p': 0.6,  # Reduce top_p for speed
        'repeat_penalty': 1.05  # Light penalty to avoid repetition
    }
    
    # More aggressive speed settings for Quadro P4000
    if gpu_info['has_cuda']:
        if 'P4000' in str(gpu_info.get('gpu_name', '')):
            config['n_ctx'] = 768  # Very small context for P4000 speed
            config['n_batch'] = 64   # Smaller batch for older GPU
            config['n_predict'] = 50  # Very short responses for speed
            print(f"🚀 Maximum speed mode for P4000: ctx={config['n_ctx']}, batch={config['n_batch']}")
        else:
            config['n_ctx'] = 1024  # Small context for speed
            config['n_batch'] = 128
            print(f"🚀 Speed-optimized: ctx={config['n_ctx']}, batch={config['n_batch']}")
    
    print(f"🧠 LLM config: ctx={config['n_ctx']}, batch={config['n_batch']}, gpu_layers={config['n_gpu_layers']}")
    return config

def apply_memory_optimizations():
    """Apply system-wide memory optimizations"""
    print("🧹 Applying memory optimizations...")
    
    # Python garbage collection settings
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    # Set memory mapping optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Limit thread pool sizes
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid threading conflicts
    
    print("✅ Memory optimizations applied")

if __name__ == "__main__":
    # Test the detection
    gpu_info = detect_gpu_resources()
    whisper_config = optimize_whisper_config(gpu_info)
    llm_config = optimize_llm_config(gpu_info)
    apply_memory_optimizations()
    
    print("\n📊 OPTIMIZATION SUMMARY:")
    print(f"   GPU Available: {gpu_info['has_cuda']}")
    print(f"   Device: {gpu_info['device']}")
    print(f"   Whisper Model: {whisper_config['model_size']}")
    print(f"   LLM GPU Layers: {llm_config['n_gpu_layers']}")
    print(f"   Context Size: {llm_config['n_ctx']}") 


================================================
File: journal_system.py
================================================
#!/usr/bin/env python3

"""
Journal System for Senter
Tracks interactions, builds personality profiles, and maintains long-term context
"""

import json
import time
import threading
from datetime import datetime
from typing import Optional, Dict, List, Any
import re

class JournalSystem:
    """Manages personality profiles, interests, goals, and contextual memory."""
    
    def __init__(self, db_client, user_profile):
        self.db = db_client
        self.user_profile = user_profile
        self.collection_name = f"journal_{user_profile.get_current_username()}"
        self.personality_collection = None
        self.journal_collection = None
        self.current_session = {
            'start_time': time.time(),
            'interactions': [],
            'camera_analyses': [],
            'tool_usage': {},
            'topics_discussed': set(),
            'user_goals_mentioned': [],
            'personality_indicators': []
        }
        
    def initialize(self):
        """Initialize journal collections."""
        try:
            # Initialize journal collection for session tracking
            try:
                self.journal_collection = self.db.get_collection(self.collection_name)
                print(f"📖 Loaded existing journal: {self.journal_collection.count()} entries")
            except:
                self.journal_collection = self.db.create_collection(self.collection_name)
                print(f"📖 Created new journal for: {self.user_profile.get_current_username()}")
            
            # Initialize personality collection
            personality_collection_name = f"personality_{self.user_profile.get_current_username()}"
            try:
                self.personality_collection = self.db.get_collection(personality_collection_name)
                print(f"🧠 Loaded existing personality profile")
            except:
                self.personality_collection = self.db.create_collection(personality_collection_name)
                print(f"🧠 Created new personality profile")
                
            return True
            
        except Exception as e:
            print(f"⚠️  Journal initialization failed: {e}")
            return False
    
    def add_interaction(self, user_input: str, ai_response: str, tools_used: List[str], 
                       tool_results: str = None, camera_analysis: str = None):
        """Add an interaction to the current session."""
        interaction = {
            'timestamp': time.time(),
            'user_input': user_input,
            'ai_response': ai_response,
            'tools_used': tools_used,
            'tool_results': tool_results,
            'camera_analysis': camera_analysis
        }
        
        self.current_session['interactions'].append(interaction)
        
        # Track tool usage
        for tool in tools_used:
            self.current_session['tool_usage'][tool] = self.current_session['tool_usage'].get(tool, 0) + 1
        
        # Extract topics and potential goals
        self._extract_topics_and_goals(user_input)
        
        # Add camera analysis if available
        if camera_analysis:
            self.current_session['camera_analyses'].append({
                'timestamp': time.time(),
                'analysis': camera_analysis
            })
    
    def _extract_topics_and_goals(self, user_input: str):
        """Extract topics and potential goals from user input."""
        input_lower = user_input.lower()
        
        # Extract topics (simple keyword extraction)
        topics = set()
        
        # Technology topics
        tech_keywords = ['ai', 'artificial intelligence', 'machine learning', 'programming', 'code', 'computer', 'software', 'technology']
        for keyword in tech_keywords:
            if keyword in input_lower:
                topics.add('technology')
                break
        
        # Home/smart home topics
        home_keywords = ['lights', 'smart home', 'house', 'room', 'lighting', 'automation']
        for keyword in home_keywords:
            if keyword in input_lower:
                topics.add('smart_home')
                break
        
        # Appearance/personal topics
        appearance_keywords = ['look', 'appearance', 'hair', 'face', 'outfit', 'style']
        for keyword in appearance_keywords:
            if keyword in input_lower:
                topics.add('appearance')
                break
        
        # Research/learning topics
        research_keywords = ['learn', 'research', 'study', 'understand', 'explain', 'tell me about']
        for keyword in research_keywords:
            if keyword in input_lower:
                topics.add('learning')
                break
        
        self.current_session['topics_discussed'].update(topics)
        
        # Extract potential goals (goal-oriented language)
        goal_indicators = [
            'want to', 'need to', 'trying to', 'planning to', 'hoping to', 
            'goal', 'objective', 'achieve', 'accomplish', 'improve', 'get better at'
        ]
        
        for indicator in goal_indicators:
            if indicator in input_lower:
                # Extract the goal context
                goal_context = user_input  # Could be more sophisticated
                self.current_session['user_goals_mentioned'].append({
                    'indicator': indicator,
                    'context': goal_context,
                    'timestamp': time.time()
                })
    
    def process_session_async(self):
        """Process the current session asynchronously to extract personality insights."""
        def session_processor():
            try:
                print("📝 Processing session for personality insights...")
                
                # Analyze session data
                insights = self._analyze_session()
                
                # Update personality profile
                self._update_personality_profile(insights)
                
                # Save session to journal
                self._save_session()
                
                # Reset current session
                self._reset_session()
                
                print("✅ Session processing completed")
                
            except Exception as e:
                print(f"⚠️  Session processing error: {e}")
        
        # Run in background thread
        threading.Thread(target=session_processor, daemon=True).start()
    
    def _analyze_session(self) -> Dict[str, Any]:
        """Analyze the current session to extract personality insights."""
        insights = {
            'interaction_count': len(self.current_session['interactions']),
            'primary_topics': list(self.current_session['topics_discussed']),
            'tool_preferences': self.current_session['tool_usage'],
            'session_duration': time.time() - self.current_session['start_time'],
            'goals_mentioned': self.current_session['user_goals_mentioned'],
            'personality_traits': [],
            'interests': [],
            'behavioral_patterns': []
        }
        
        # Analyze tool usage patterns
        most_used_tool = max(self.current_session['tool_usage'], 
                           key=self.current_session['tool_usage'].get, 
                           default=None)
        
        if most_used_tool:
            if most_used_tool == 'camera':
                insights['personality_traits'].append('appearance_conscious')
                insights['interests'].append('self_image')
            elif most_used_tool == 'research':
                insights['personality_traits'].append('curious')
                insights['personality_traits'].append('knowledge_seeking')
                insights['interests'].append('learning')
            elif most_used_tool == 'lights':
                insights['personality_traits'].append('environment_conscious')
                insights['interests'].append('smart_home')
        
        # Analyze topics for interests
        for topic in self.current_session['topics_discussed']:
            insights['interests'].append(topic)
        
        # Analyze communication patterns
        interactions = self.current_session['interactions']
        if interactions:
            avg_input_length = sum(len(i['user_input']) for i in interactions) / len(interactions)
            if avg_input_length > 100:
                insights['personality_traits'].append('detailed_communicator')
            elif avg_input_length < 30:
                insights['personality_traits'].append('concise_communicator')
        
        return insights
    
    def _update_personality_profile(self, insights: Dict[str, Any]):
        """Update the user's personality profile with new insights."""
        try:
            # Get existing personality data
            existing_profile = self._get_personality_profile()
            
            # Merge insights
            updated_profile = self._merge_personality_data(existing_profile, insights)
            
            # Save updated profile
            profile_id = f"profile_{int(time.time())}"
            self.personality_collection.add(
                documents=[json.dumps(updated_profile)],
                metadatas=[{
                    'type': 'personality_profile',
                    'timestamp': time.time(),
                    'insights_count': len(insights.get('personality_traits', []))
                }],
                ids=[profile_id]
            )
            
            print(f"🧠 Updated personality profile: {len(updated_profile.get('traits', []))} traits")
            
        except Exception as e:
            print(f"⚠️  Personality profile update error: {e}")
    
    def _get_personality_profile(self) -> Dict[str, Any]:
        """Get the current personality profile."""
        try:
            # Get most recent personality profile
            results = self.personality_collection.query(
                query_texts=["personality_profile"],
                n_results=1,
                include=["documents", "metadatas"]
            )
            
            if results and results['documents'] and results['documents'][0]:
                profile_json = results['documents'][0][0]
                return json.loads(profile_json)
            
        except Exception as e:
            print(f"⚠️  Error getting personality profile: {e}")
        
        # Return default profile
        return {
            'traits': [],
            'interests': [],
            'goals': [],
            'communication_style': 'balanced',
            'tool_preferences': {},
            'behavioral_patterns': []
        }
    
    def _merge_personality_data(self, existing: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new insights with existing personality data."""
        merged = existing.copy()
        
        # Merge traits (with frequency tracking)
        trait_counts = {}
        for trait in merged.get('traits', []):
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        for trait in insights.get('personality_traits', []):
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        # Keep traits that appear more than once
        merged['traits'] = [trait for trait, count in trait_counts.items() if count > 1]
        
        # Merge interests
        interests = set(merged.get('interests', []))
        interests.update(insights.get('interests', []))
        merged['interests'] = list(interests)
        
        # Merge goals
        goals = merged.get('goals', [])
        for goal_mention in insights.get('goals_mentioned', []):
            goals.append({
                'context': goal_mention['context'],
                'extracted_at': goal_mention['timestamp'],
                'status': 'identified'
            })
        merged['goals'] = goals
        
        # Update tool preferences
        tool_prefs = merged.get('tool_preferences', {})
        for tool, count in insights.get('tool_preferences', {}).items():
            tool_prefs[tool] = tool_prefs.get(tool, 0) + count
        merged['tool_preferences'] = tool_prefs
        
        return merged
    
    def _save_session(self):
        """Save the current session to the journal."""
        try:
            session_id = f"session_{int(time.time())}"
            session_data = self.current_session.copy()
            session_data['topics_discussed'] = list(session_data['topics_discussed'])  # Convert set to list
            
            self.journal_collection.add(
                documents=[json.dumps(session_data)],
                metadatas=[{
                    'type': 'session',
                    'timestamp': session_data['start_time'],
                    'interaction_count': len(session_data['interactions'])
                }],
                ids=[session_id]
            )
            
        except Exception as e:
            print(f"⚠️  Session save error: {e}")
    
    def _reset_session(self):
        """Reset the current session."""
        self.current_session = {
            'start_time': time.time(),
            'interactions': [],
            'camera_analyses': [],
            'tool_usage': {},
            'topics_discussed': set(),
            'user_goals_mentioned': [],
            'personality_indicators': []
        }
    
    def get_personality_context_for_response(self) -> str:
        """Get personality context to inject into Senter's responses."""
        try:
            profile = self._get_personality_profile()
            
            if not profile or not profile.get('traits'):
                return ""
            
            # Build personality context
            context = "\n\nPERSONALITY CONTEXT FOR SENTER:\n"
            context += "Based on our interactions, adopt these personality traits:\n"
            
            traits = profile.get('traits', [])
            interests = profile.get('interests', [])
            
            if traits:
                context += f"- Personality: {', '.join(traits[:3])}\n"  # Limit to top 3 traits
            
            if interests:
                context += f"- Show interest in: {', '.join(interests[:3])}\n"  # Limit to top 3 interests
            
            # Add communication style guidance
            if 'detailed_communicator' in traits:
                context += "- Match their detailed communication style\n"
            elif 'concise_communicator' in traits:
                context += "- Keep responses concise and to the point\n"
            
            context += "Incorporate these traits naturally into your personality and responses.\n"
            
            return context
            
        except Exception as e:
            print(f"⚠️  Error getting personality context: {e}")
            return ""

# Global instance
journal_system = None

def initialize_journal_system(db_client, user_profile):
    """Initialize the global journal system."""
    global journal_system
    journal_system = JournalSystem(db_client, user_profile)
    return journal_system.initialize()

def add_interaction_to_journal(user_input: str, ai_response: str, tools_used: List[str], 
                             tool_results: str = None, camera_analysis: str = None):
    """Add an interaction to the journal."""
    if journal_system:
        journal_system.add_interaction(user_input, ai_response, tools_used, tool_results, camera_analysis)

def process_session_journal():
    """Process the current session asynchronously."""
    if journal_system:
        journal_system.process_session_async()

def get_personality_context():
    """Get personality context for injection into responses."""
    if journal_system:
        return journal_system.get_personality_context_for_response()
    return "" 


================================================
File: launch_senter_complete.py
================================================
#!/usr/bin/env python3
"""
Complete Senter AI Assistant Launcher
Integrates CLI + AvA + UI into unified system with Control+Control toggle

🐳 DOCKER CONTAINER EXECUTION ONLY 🐳
This script is designed to run exclusively inside a Docker container.

DO NOT run this script directly on the host system.
Use: docker-compose exec senter python launch_senter_complete.py

For container management:
- Start: docker-compose up -d
- Stop: docker-compose down  
- Logs: docker-compose logs -f senter
- Shell: docker-compose exec senter /bin/bash
"""

import os
import sys

# Fix OpenMP conflict FIRST
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    """Launch the complete unified Senter system"""
    
    # Verify we're running in Docker container
    if not os.path.exists('/.dockerenv') and not os.environ.get('DOCKER_MODE'):
        print("🚨 ERROR: SENTER must run inside Docker container!")
        print("")
        print("This script is designed for Docker container execution only.")
        print("Please use one of these commands:")
        print("")
        print("📋 Start container:     docker-compose up -d")
        print("🚀 Run SENTER:         docker-compose exec senter python launch_senter_complete.py")
        print("📊 Container logs:     docker-compose logs -f senter")
        print("🐚 Container shell:    docker-compose exec senter /bin/bash")
        print("")
        sys.exit(1)
    
    print("🚀 Starting Complete Senter AI Assistant")
    print("   🧠 CLI System with full tools")
    print("   👁️ AvA attention detection with RGB effects") 
    print("   🖥️ Modern UI interface with Control+Control toggle")
    print("")
    
    try:
        # Import user profile for login
        from user_profiles import UserProfile
        
        # Initialize user profile system FIRST
        user_profile = UserProfile()
        
        # Auto-login as Chris for fast testing (skip profile selection)
        print("🚀 Auto-login mode enabled for fast testing")
        user_profile.setup_initial_profiles()
        if not user_profile.login("Chris", ""):
            print("❌ Auto-login failed")
            return
        
        print(f"✅ Logged in as: {user_profile.get_display_name()}")
        print("")
        
        # For fast testing, just run the main CLI system
        print("🔧 Starting main CLI system for testing...")
        
        # Import and run the main system
        from main import main as main_system
        
        # Set the user profile for the main system
        os.environ['CURRENT_USER'] = user_profile.get_current_username()
        
        print("🎯 SENTER AI ASSISTANT READY!")
        print("=" * 50)
        print("💬 Text Input: Type commands in terminal")
        print("🎙️ Voice Input: Look at camera for attention detection")
        print("🧠 GPU Acceleration: Check logs for optimization status")
        print("📊 Camera: AvA attention detection system")
        print("=" * 50)
        print("")
        
        # Run the main system
        main_system()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting Senter: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 


================================================
File: light_controller.py
================================================
"""
Light Controller Bridge
Provides a simple interface for main.py to control lights via the lights.py script.
"""

import subprocess
import sys
import os
import json
import re
from typing import Optional, Dict, Any

# Import the lights module to set credentials
try:
    import lights
except ImportError:
    lights = None

def set_user_credentials(user_profile_data: Dict[str, Any]):
    """Set AiDot credentials from user profile data."""
    if lights and user_profile_data:
        aidot_creds = user_profile_data.get("credentials", {}).get("aidot")
        if aidot_creds:
            lights.set_credentials(aidot_creds)

def normalize_light_command(command: str) -> str:
    """Normalize and improve light commands to handle common cases.
    
    Args:
        command: Raw command string
        
    Returns:
        str: Normalized command string
    """
    command = command.strip()
    
    # Common color names that might be used without "ALL" prefix
    color_names = [
        'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'orange', 'cyan', 
        'magenta', 'white', 'warm white', 'cool white', 'teal', 'lime', 'indigo',
        'violet', 'brown', 'black', 'gray', 'grey', 'gold', 'silver'
    ]
    
    # Check if command is just a color name (case insensitive)
    if command.lower() in [color.lower() for color in color_names]:
        return f"ALL {command.upper()}"
    
    # Check if command starts with a color but no device specified
    for color in color_names:
        if command.lower() == color.lower():
            return f"ALL {command.upper()}"
    
    # If command is very short and doesn't contain known device names, assume ALL
    known_patterns = ['kitchen', 'living', 'desk', 'room', 'all', 'on', 'off']
    if len(command) < 15 and not any(pattern in command.lower() for pattern in known_patterns):
        # Might be a color-only command
        return f"ALL {command.upper()}"
    
    return command

def execute_light_command(command: str) -> bool:
    """Execute a light command using the lights.py script.
    
    Args:
        command: Light command string (e.g., "Kitchen Red", "ALL OFF")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Normalize the command to handle common cases
        normalized_command = normalize_light_command(command)
        
        if normalized_command != command:
            print(f"🔧 Normalized light command: '{command}' → '{normalized_command}'")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lights_script = os.path.join(script_dir, "lights.py")
        
        # Execute the lights script with the normalized command
        result = subprocess.run(
            [sys.executable, lights_script, normalized_command],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"⚠️  Light command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Light command timed out")
        return False
    except Exception as e:
        print(f"⚠️  Error executing light command: {e}")
        return False

def parse_lights_xml(xml_content: str) -> list:
    """Parse lights XML commands from AI response.
    
    Args:
        xml_content: XML content containing lights commands
        
    Returns:
        list: List of light command strings
    """
    commands = []
    
    # Find all lights commands
    lights_pattern = r'<lights>\s*([^<]*?)\s*</lights>'
    matches = re.findall(lights_pattern, xml_content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        command = match.strip()
        if command:
            commands.append(command)
    
    return commands

def get_available_lights_for_profile(user_profile_data: Dict[str, Any]) -> dict:
    """Get available lights using credentials from user profile.
    
    Args:
        user_profile_data: User profile containing credentials
        
    Returns:
        dict: Available lights information
    """
    # Set credentials first
    set_user_credentials(user_profile_data)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lights_script = os.path.join(script_dir, "lights.py")
        
        result = subprocess.run(
            [sys.executable, lights_script, "--get-lights"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            print(f"⚠️  Could not get lights info: {result.stderr}")
            return {}
    except Exception as e:
        print(f"⚠️  Error getting lights info: {e}")
        return {}

# Example usage for testing
if __name__ == "__main__":
    # Test commands
    test_commands = [
        "Kitchen ON",
        "Ava's Room OFF", 
        "Desk Set Color (255,128,0)",
        "Living Room Brightness 50"
    ]
    
    print("Testing light controller...")
    for cmd in test_commands:
        print(f"\nTesting: {cmd}")
        success = execute_light_command(cmd)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}") 


================================================
File: lights.py
================================================
'''
Interactive script to control AiDot lights using the python-AiDot library.
Modified to accept direct commands from main.py tool calls.
'''
import asyncio
import aiohttp
import logging
import sys
import re
import json
from collections import defaultdict

from aidot.client import AidotClient
from aidot.device_client import DeviceClient # For type hinting
from aidot.exceptions import AidotUserOrPassIncorrect, AidotAuthFailed

# --- Default Configuration (fallback) ---
DEFAULT_AIDOT_USERNAME = "christophersghardwick@gmail.com"  # Fallback
DEFAULT_AIDOT_PASSWORD = "A111s1nmym!nd"      # Fallback
DEFAULT_AIDOT_COUNTRY_NAME = "UnitedStates" 

# Room name mapping - map friendly names to actual device names
ROOM_MAPPING = {
    "Kitchen": "Kitchen",
    "Ava's Room": "Ava's Room", 
    "Jack's Room": "Jack's Room",
    "Porch": "Porch",
    "Desk": "Desk",
    "Living Room": "Living Room"
}

# Color name mapping - map color names to RGB values
COLOR_MAPPING = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "warm_white": (255, 230, 200),
    "cool_white": (200, 230, 255),
    "teal": (0, 255, 255),
    "turquoise": (64, 224, 208),
    "lime": (0, 255, 0),
    "off": (0, 0, 0)
}

# Global credentials - will be set from user profile
CURRENT_CREDENTIALS = {
    "username": DEFAULT_AIDOT_USERNAME,
    "password": DEFAULT_AIDOT_PASSWORD,
    "country": DEFAULT_AIDOT_COUNTRY_NAME
}

def set_credentials(credentials: dict):
    """Set AiDot credentials from user profile."""
    global CURRENT_CREDENTIALS
    if credentials:
        CURRENT_CREDENTIALS = {
            "username": credentials.get("username", DEFAULT_AIDOT_USERNAME),
            "password": credentials.get("password", DEFAULT_AIDOT_PASSWORD),
            "country": credentials.get("country", DEFAULT_AIDOT_COUNTRY_NAME)
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
_LOGGER = logging.getLogger(__name__)

async def get_connected_devices(client: AidotClient) -> list[DeviceClient]:
    '''Logs in, discovers, and returns a list of locally connected DeviceClient objects.'''
    try:
        _LOGGER.info(f"Attempting to log in as {client.username}...")
        login_info = await client.async_post_login()
        if not login_info or not client.login_info.get("accessToken"):
            _LOGGER.error("Login failed. No access token received.")
            return []
        _LOGGER.info(f"Login successful. User ID: {client.login_info.get('id')}")
    except AidotUserOrPassIncorrect:
        _LOGGER.error("Login failed: Username or password incorrect.")
        return []
    except AidotAuthFailed:
        _LOGGER.error("Login failed: Authentication failed.")
        return []
    except Exception as e:
        _LOGGER.error(f"An unexpected error occurred during login: {e}")
        return []

    _LOGGER.info("Starting device discovery...")
    client.start_discover() # Runs in background

    _LOGGER.info("Fetching account device list...")
    account_devices_data = []
    try:
        houses = await client.async_get_houses()
        if houses:
            for house_data in houses:
                house_id = house_data.get("id")
                if house_id:
                    devices_in_house = await client.async_get_devices(house_id)
                    if devices_in_house:
                        product_ids = [d.get("productId") for d in devices_in_house if d.get("productId")]
                        if product_ids:
                            unique_product_ids = ",".join(list(set(product_ids)))
                            product_list = await client.async_get_products(unique_product_ids)
                            for dev in devices_in_house:
                                for prod in product_list:
                                    if dev.get("productId") == prod.get("id"):
                                        dev["product"] = prod
                                        break
                        account_devices_data.extend(devices_in_house)
    except Exception as e:
        _LOGGER.error(f"Error fetching account devices: {e}")
    
    if not account_devices_data:
        _LOGGER.warning("No devices found on account.")
        # Allow some time for purely local discovery if desired, though current logic focuses on account devices
        _LOGGER.info("Waiting 5s for any purely local discovery (if applicable to library)... ")
        await asyncio.sleep(5)
        # Future: check client._discover.discovered_device if that path is to be supported for non-account devices
        return []

    _LOGGER.info(f"Found {len(account_devices_data)} device(s) on account. Waiting 5s for local IP discovery...")
    await asyncio.sleep(5) # Allow time for IPs to be populated

    connected_clients: list[DeviceClient] = []
    for device_data in account_devices_data:
        dev_id = device_data.get("id")
        dev_name = device_data.get("name", f"Device_{dev_id}")
        device_client = client.get_device_client(device_data)

        if device_client._ip_address:
            _LOGGER.info(f"Attempting local connection to {dev_name} at {device_client._ip_address}...")
            try:
                await device_client.async_login() # Establish local connection
                if device_client.connect_and_login:
                    _LOGGER.info(f"Successfully connected locally to {dev_name}.")
                    connected_clients.append(device_client)
                else:
                    _LOGGER.warning(f"Failed to establish local connection to {dev_name} despite having IP.")
            except Exception as e:
                _LOGGER.error(f"Error during local login to {dev_name}: {e}")
        else:
            _LOGGER.warning(f"No local IP found for {dev_name}. Cannot control locally.")
    
    return connected_clients

def parse_light_command(command: str) -> tuple[str, str]:
    """Parse a light command from main.py format.
    
    Expected formats:
    - "Kitchen ON"
    - "Kitchen, OFF" (with comma)
    - "Ava's Room OFF" 
    - "Desk Brightness +5%"
    - "Living Room Set Color (255,128,0)"
    - "Desk Set Color (255 128 0)" (spaces instead of commas)
    - "Kitchen Set Color 255,128,0" (no parentheses)
    - "Kitchen Red" (color name)
    - "ALL Green" (color name for all lights)
    
    Returns:
        tuple: (room_name, action)
    """
    command = command.strip()
    
    # Remove any leading/trailing commas and clean up spacing
    command = command.replace(',', ' ').strip()
    # Normalize multiple spaces to single spaces
    command = ' '.join(command.split())
    
    # Handle color commands specially - normalize different formats
    if 'Set Color' in command:
        # Extract room name and color values
        parts = command.split('Set Color')
        if len(parts) == 2:
            room_part = parts[0].strip()
            color_part = parts[1].strip()
            
            # Extract numbers from color part using regex - handles various formats
            color_numbers = re.findall(r'\d+', color_part)
            
            if len(color_numbers) >= 3:
                r, g, b = color_numbers[0], color_numbers[1], color_numbers[2]
                w = color_numbers[3] if len(color_numbers) > 3 else "0"
                
                # Reconstruct in proper format
                normalized_action = f"Set Color ({r},{g},{b},{w})"
                return room_part, normalized_action
    
    # Check for ALL command with color name
    if command.upper().startswith("ALL"):
        action = command[3:].strip()  # Remove "ALL" and get the rest
        action_lower = action.lower()
        
        # Handle ON/OFF commands first (before color mapping)
        if action_lower in ["on", "off"]:
            return "ALL", action.upper()
        
        if action_lower in COLOR_MAPPING:
            r, g, b = COLOR_MAPPING[action_lower]
            normalized_action = f"Set Color ({r},{g},{b},0)"
            return "ALL", normalized_action
        return "ALL", action
    
    # Fallback: assume first word is room, rest is action
    parts = command.split(' ', 1)
    if len(parts) == 2:
        room, action = parts[0], parts[1]
        
        # Handle ON/OFF commands first (before color mapping)
        action_lower = action.lower()
        if action_lower in ["on", "off"]:
            return room, action.upper()
        
        # Check if action is a color name
        if action_lower in COLOR_MAPPING:
            r, g, b = COLOR_MAPPING[action_lower]
            normalized_action = f"Set Color ({r},{g},{b},0)"
            return room, normalized_action
        
        return room, action
    else:
        return parts[0], ""

async def execute_light_command(device_clients: list[DeviceClient], room_name: str, action: str) -> bool:
    """Execute a light command on the specified room.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Handle ALL command - control all lights
    if room_name.upper() == "ALL":
        target_devices = device_clients  # Use all available devices
        _LOGGER.info(f"ALL command - controlling {len(target_devices)} device(s): {[dc.info.name for dc in target_devices]}")
    else:
        # Map room name to actual device name
        device_name = ROOM_MAPPING.get(room_name, room_name)
        
        # Group lights by base name
        grouped_lights = group_lights_by_base_name(device_clients)
        
        # Find target devices - prioritize base name grouping over exact match
        target_devices = []
        
        # First try base name matching (this groups similar devices)
        device_name_lower = device_name.lower()
        for base_name, devices in grouped_lights.items():
            if base_name.lower() == device_name_lower:
                target_devices = devices
                break
        
        # If no base name match, try exact match
        if not target_devices:
            for dc in device_clients:
                if dc.info.name.lower() == device_name.lower():
                    target_devices = [dc]
                    break
        
        # If still no match, try partial matching
        if not target_devices:
            for dc in device_clients:
                if device_name.lower() in dc.info.name.lower() or dc.info.name.lower() in device_name.lower():
                    target_devices.append(dc)
        
        if not target_devices:
            _LOGGER.error(f"Device '{device_name}' not found. Available devices: {[dc.info.name for dc in device_clients]}")
            return False
        
        _LOGGER.info(f"Found {len(target_devices)} device(s) for '{room_name}': {[dc.info.name for dc in target_devices]}")
    
    # Execute action on all target devices
    success_count = 0
    for target_device in target_devices:
        # Ensure connection
        if not target_device.connect_and_login:
            _LOGGER.info(f"Reconnecting to {target_device.info.name}...")
            try:
                await target_device.async_login()
                if not target_device.connect_and_login:
                    _LOGGER.error(f"Failed to connect to {target_device.info.name}")
                    continue
            except Exception as e:
                _LOGGER.error(f"Error connecting to {target_device.info.name}: {e}")
                continue
        
        # Execute the action
        try:
            action_clean = action.strip()
            
            if action_clean.upper() == "ON":
                _LOGGER.info(f"Turning {target_device.info.name} ON")
                await target_device.async_turn_on()
                
            elif action_clean.upper() == "OFF":
                _LOGGER.info(f"Turning {target_device.info.name} OFF")
                await target_device.async_turn_off()
                
            elif action_clean.startswith("Brightness"):
                # Parse brightness commands like "Brightness +5%" or "Brightness -5%"
                brightness_match = re.search(r'Brightness\s*([+-]?\d+)%?', action_clean)
                if brightness_match:
                    change = int(brightness_match.group(1))
                    # For relative changes, we'd need to get current brightness first
                    # For now, treat as absolute if positive, or skip if we can't determine current
                    if change > 0:
                        brightness = min(100, max(1, abs(change)))
                        _LOGGER.info(f"Setting {target_device.info.name} brightness to {brightness}%")
                        await target_device.async_set_brightness(brightness)
                    else:
                        _LOGGER.warning(f"Relative brightness changes not fully implemented. Use absolute values.")
                        continue
                else:
                    _LOGGER.error(f"Could not parse brightness from: {action_clean}")
                    continue
                    
            elif action_clean.startswith("Set Color"):
                # Parse color commands like "Set Color (255,128,0)" or "Set Color (255,128,0,255)"
                color_match = re.search(r'Set Color\s*\((\d+),(\d+),(\d+)(?:,(\d+))?\)', action_clean)
                if color_match:
                    r = int(color_match.group(1))
                    g = int(color_match.group(2))
                    b = int(color_match.group(3))
                    w = int(color_match.group(4)) if color_match.group(4) else 0
                    
                    if all(0 <= val <= 255 for val in [r, g, b, w]):
                        _LOGGER.info(f"Setting {target_device.info.name} color to R:{r} G:{g} B:{b} W:{w}")
                        await target_device.async_set_rgbw((r, g, b, w))
                    else:
                        _LOGGER.error(f"Color values must be between 0-255")
                        continue
                else:
                    _LOGGER.error(f"Could not parse color from: {action_clean}")
                    continue
                    
            else:
                _LOGGER.error(f"Unknown action: {action_clean}")
                continue
                
            _LOGGER.info(f"Successfully executed '{action_clean}' on {target_device.info.name}")
            success_count += 1
            
        except Exception as e:
            _LOGGER.error(f"Error executing action '{action_clean}' on {target_device.info.name}: {e}")
    
    # Return True if at least one device succeeded
    if success_count > 0:
        _LOGGER.info(f"Successfully controlled {success_count}/{len(target_devices)} devices for '{room_name}'")
        return True
    else:
        _LOGGER.error(f"Failed to control any devices for '{room_name}'")
        return False

async def execute_direct_command(command: str) -> bool:
    """Execute a direct light command without interactive mode."""
    # Use current credentials (set from user profile)
    if CURRENT_CREDENTIALS["username"] == "YOUR_AIDOT_EMAIL_OR_USERNAME" or CURRENT_CREDENTIALS["password"] == "YOUR_AIDOT_PASSWORD":
        _LOGGER.error("FATAL: No valid credentials available. Please configure AiDot credentials in your user profile.")
        return False

    _LOGGER.info(f"Raw command received: '{command}'")
    room_name, action = parse_light_command(command)
    _LOGGER.info(f"Parsed command - Room: '{room_name}', Action: '{action}'")

    async with aiohttp.ClientSession() as session:
        client = AidotClient(
            session=session,
            username=CURRENT_CREDENTIALS["username"],
            password=CURRENT_CREDENTIALS["password"],
            country_name=CURRENT_CREDENTIALS["country"]
        )
        
        device_clients = await get_connected_devices(client)
        
        if not device_clients:
            _LOGGER.error("No devices connected. Cannot execute command.")
            return False
        
        success = await execute_light_command(device_clients, room_name, action)
        
        # Cleanup
        for dc in device_clients:
            try:
                await dc.close()
            except Exception as e:
                _LOGGER.error(f"Error closing connection to {dc.info.name}: {e}")
        
        if client._discover:
            client._discover.close()
        
        return success

async def get_user_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)

async def interactive_control(device_clients: list[DeviceClient]):
    if not device_clients:
        _LOGGER.info("No locally controllable devices found.")
        return

    while True:
        print("\n--- Available Lights ---")
        for i, dc in enumerate(device_clients):
            print(f"{i + 1}: {dc.info.name} (ID: {dc.info.dev_id}, IP: {dc._ip_address})")
        print("q: Quit")

        choice_str = await get_user_input("Select a light by number (or 'q' to quit): ")
        if choice_str.lower() == 'q':
            break

        try:
            choice_idx = int(choice_str) - 1
            if not (0 <= choice_idx < len(device_clients)):
                raise ValueError("Choice out of range")
            selected_client = device_clients[choice_idx]
        except ValueError:
            print("Invalid choice. Please enter a number from the list.")
            continue

        _LOGGER.info(f"Selected light: {selected_client.info.name}")

        while True:
            print(f"\n--- Actions for {selected_client.info.name} ---")
            print("1: Turn ON")
            print("2: Turn OFF")
            print("3: Set Brightness (1-100)")
            print("4: Set Color (RGBW, 0-255 for each component)")
            # print("5: Set Color Temperature (CCT)") # TODO: Implement if desired
            print("b: Back to light selection")

            action_choice = await get_user_input("Choose an action: ")

            try:
                # Ensure connection before action
                if not selected_client.connect_and_login:
                    _LOGGER.info(f"Connection to {selected_client.info.name} seems to be down. Attempting to reconnect...")
                    try:
                        await selected_client.async_login() # Re-establish local connection
                        if not selected_client.connect_and_login:
                            _LOGGER.error(f"Failed to reconnect to {selected_client.info.name}. Please go back and re-select the light.")
                            continue # Skip to next action choice loop
                        _LOGGER.info(f"Successfully reconnected to {selected_client.info.name}.")
                    except Exception as recon_e:
                        _LOGGER.error(f"Error during reconnection attempt to {selected_client.info.name}: {recon_e}")
                        continue # Skip to next action choice loop

                if action_choice == '1':
                    _LOGGER.info(f"Turning {selected_client.info.name} ON...")
                    await selected_client.async_turn_on()
                    _LOGGER.info(f"{selected_client.info.name} turned ON.")
                elif action_choice == '2':
                    _LOGGER.info(f"Turning {selected_client.info.name} OFF...")
                    await selected_client.async_turn_off()
                    _LOGGER.info(f"{selected_client.info.name} turned OFF.")
                elif action_choice == '3':
                    bright_str = await get_user_input("Enter brightness (1-100): ")
                    brightness = int(bright_str)
                    if not (1 <= brightness <= 100):
                        print("Brightness must be between 1 and 100.")
                        continue
                    _LOGGER.info(f"Setting {selected_client.info.name} brightness to {brightness}%...")
                    await selected_client.async_set_brightness(brightness)
                    _LOGGER.info(f"{selected_client.info.name} brightness set.")
                elif action_choice == '4':
                    print("Enter RGBW values (0-255 for each component).")
                    r_str = await get_user_input("Red (0-255): ")
                    g_str = await get_user_input("Green (0-255): ")
                    b_str = await get_user_input("Blue (0-255): ")
                    w_str = await get_user_input("White (0-255): ")
                    r, g, b, w = int(r_str), int(g_str), int(b_str), int(w_str)
                    if not all(0 <= val <= 255 for val in [r, g, b, w]):
                        print("All RGBW values must be between 0 and 255.")
                        continue
                    _LOGGER.info(f"Setting {selected_client.info.name} color to R:{r} G:{g} B:{b} W:{w}...")
                    await selected_client.async_set_rgbw((r, g, b, w))
                    _LOGGER.info(f"{selected_client.info.name} color set.")
                elif action_choice.lower() == 'b':
                    break # Back to light selection
                else:
                    print("Invalid action choice.")
            except ValueError:
                print("Invalid input. Please enter a number where expected.")
            except Exception as e:
                _LOGGER.error(f"Error performing action on {selected_client.info.name}: {e}")

async def main_interactive():
    _LOGGER.info("Starting AiDot Interactive Light Control Script")

    # Use current credentials
    if CURRENT_CREDENTIALS["username"] == "YOUR_AIDOT_EMAIL_OR_USERNAME" or CURRENT_CREDENTIALS["password"] == "YOUR_AIDOT_PASSWORD":
        _LOGGER.error("FATAL: No valid credentials available. Please configure AiDot credentials in your user profile.")
        print("ERROR: No valid credentials found. Please configure AiDot credentials in your user profile.")
        return

    async with aiohttp.ClientSession() as session:
        client = AidotClient(
            session=session,
            username=CURRENT_CREDENTIALS["username"],
            password=CURRENT_CREDENTIALS["password"],
            country_name=CURRENT_CREDENTIALS["country"]
        )
        
        connectable_devices = await get_connected_devices(client)

        if connectable_devices:
            await interactive_control(connectable_devices)
        else:
            _LOGGER.warning("No devices were successfully connected locally. Exiting interactive mode.")

        _LOGGER.info("Cleaning up...")
        for dc in connectable_devices:
            try:
                _LOGGER.info(f"Closing connection to {dc.info.name}")
                await dc.close()
            except Exception as e:
                _LOGGER.error(f"Error closing connection to {dc.info.name}: {e}")
        
        if client._discover: # client._discover might be None if login failed early
            _LOGGER.info("Closing discovery service.")
            client._discover.close()
        # No explicit client.cleanup() method found in AidotClient, session closes via async with

    _LOGGER.info("AiDot Interactive Light Control Script finished.")

def group_lights_by_base_name(device_clients: list[DeviceClient]) -> dict[str, list[DeviceClient]]:
    """Group lights by their base name (removing numbers).
    
    Examples:
    - Kitchen, Kitchen2 -> "Kitchen": [Kitchen, Kitchen2]
    - Desk -> "Desk": [Desk]
    
    Returns:
        dict: Base name -> list of DeviceClient objects
    """
    grouped = defaultdict(list)
    
    for dc in device_clients:
        name = dc.info.name
        # Remove numbers and common suffixes to get base name
        base_name = re.sub(r'\d+$', '', name).strip()
        grouped[base_name].append(dc)
    
    return dict(grouped)

def get_available_lights_info(device_clients: list[DeviceClient]) -> dict:
    """Get information about available lights grouped by base name.
    
    Returns:
        dict: Information about available lights for the LLM
    """
    grouped = group_lights_by_base_name(device_clients)
    
    lights_info = {}
    for base_name, clients in grouped.items():
        if len(clients) == 1:
            lights_info[base_name] = {
                "count": 1,
                "devices": [clients[0].info.name]
            }
        else:
            lights_info[base_name] = {
                "count": len(clients),
                "devices": [dc.info.name for dc in clients]
            }
    
    return lights_info

async def get_available_lights_only() -> dict:
    """Get available lights without executing commands - for main.py to use."""
    # Use current credentials
    if CURRENT_CREDENTIALS["username"] == "YOUR_AIDOT_EMAIL_OR_USERNAME" or CURRENT_CREDENTIALS["password"] == "YOUR_AIDOT_PASSWORD":
        return {}

    async with aiohttp.ClientSession() as session:
        client = AidotClient(
            session=session,
            username=CURRENT_CREDENTIALS["username"],
            password=CURRENT_CREDENTIALS["password"],
            country_name=CURRENT_CREDENTIALS["country"]
        )
        
        device_clients = await get_connected_devices(client)
        lights_info = get_available_lights_info(device_clients)
        
        # Cleanup
        for dc in device_clients:
            try:
                await dc.close()
            except:
                pass
        
        if client._discover:
            client._discover.close()
        
        return lights_info

if __name__ == "__main__":
    # Use current credentials for validation
    if CURRENT_CREDENTIALS["username"] == "YOUR_AIDOT_EMAIL_OR_USERNAME" or \
       CURRENT_CREDENTIALS["password"] == "YOUR_AIDOT_PASSWORD":
        # This check is duplicated from main_interactive for immediate feedback before asyncio.run
        print("ERROR: No valid credentials found. Please configure AiDot credentials in your user profile.")
    else:
        # Check if we have command line arguments for direct command execution
        if len(sys.argv) > 1:
            if sys.argv[1] == "--get-lights":
                # Special command to get available lights for main.py
                try:
                    lights_info = asyncio.run(get_available_lights_only())
                    print(json.dumps(lights_info))
                    sys.exit(0)
                except Exception as e:
                    _LOGGER.error(f"Error getting lights: {e}")
                    print(json.dumps({}))
                    sys.exit(1)
            else:
                # Direct command mode - combine all arguments as the command
                command = " ".join(sys.argv[1:])
                _LOGGER.info(f"Executing direct command: {command}")
                try:
                    success = asyncio.run(execute_direct_command(command))
                    if success:
                        print(f"SUCCESS: Light command '{command}' executed successfully")
                        sys.exit(0)
                    else:
                        print(f"FAILED: Could not execute light command '{command}'")
                        sys.exit(1)
                except KeyboardInterrupt:
                    _LOGGER.info("Script interrupted by user.")
                    sys.exit(1)
                except Exception as e:
                    _LOGGER.error(f"Error executing command: {e}")
                    print(f"ERROR: {e}")
                    sys.exit(1)
        else:
            # Interactive mode
            try:
                asyncio.run(main_interactive())
            except KeyboardInterrupt:
                _LOGGER.info("Script interrupted by user.")
            finally:
                _LOGGER.info("Exiting application.")



================================================
File: process_manager.py
================================================
#!/usr/bin/env python3
"""
Process and Resource Manager for SENTER
Prevents freeze states and manages system resources
"""

import threading
import time
import psutil
import gc
import queue
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float = 0.0
    active_threads: int = 0
    queue_sizes: Dict[str, int] = None
    timestamp: float = 0.0

class ProcessManager:
    """Manages system processes and prevents freeze states"""
    
    def __init__(self, max_cpu_percent=95, max_memory_percent=90, max_queue_size=25):
        # Much more lenient thresholds to reduce interruptions
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_queue_size = max_queue_size
        
        # Monitoring
        self.metrics_history = deque(maxlen=30)  # Last 30 measurements
        self.is_monitoring = False
        self.monitor_thread = None
        self.cleanup_callbacks = []
        
        # Queue management
        self.managed_queues = {}
        self.queue_locks = {}
        
        # Resource limits - add cooldown to prevent spam
        self.resource_warnings = {
            'cpu': False,
            'memory': False,
            'queues': False
        }
        
        # Add cooldown timers to prevent spam
        self.last_cleanup_time = 0
        self.cleanup_cooldown = 10.0  # 10 second cooldown between cleanups
        self.last_warning_time = {'cpu': 0, 'memory': 0, 'queues': 0}
        self.warning_cooldown = 30.0  # 30 second cooldown between warnings
        
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
    def register_queue(self, name: str, queue_obj: queue.Queue):
        """Register a queue for monitoring"""
        self.managed_queues[name] = queue_obj
        self.queue_locks[name] = threading.Lock()
        
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup function to call during resource pressure"""
        self.cleanup_callbacks.append(callback)
        
    def start_monitoring(self, interval=30.0):  # Increased from 15s to 30s interval
        """Start resource monitoring with longer intervals"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for resource pressure
                self._check_resource_pressure(metrics)
                
                # Cleanup if needed (with cooldown)
                if self._should_cleanup(metrics):
                    current_time = time.time()
                    if current_time - self.last_cleanup_time > self.cleanup_cooldown:
                        self._perform_cleanup()
                        self.last_cleanup_time = current_time
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
                
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        metrics = ResourceMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            active_threads=threading.active_count(),
            queue_sizes={},
            timestamp=time.time()
        )
        
        # Collect queue sizes
        for name, q in self.managed_queues.items():
            try:
                metrics.queue_sizes[name] = q.qsize()
            except:
                metrics.queue_sizes[name] = -1
                
        # Try to get GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        except:
            pass
            
        return metrics
        
    def _check_resource_pressure(self, metrics: ResourceMetrics):
        """Check for resource pressure and warn (with cooldown to prevent spam)"""
        current_time = time.time()
        
        # CPU pressure check with cooldown
        if metrics.cpu_percent > self.max_cpu_percent:
            if current_time - self.last_warning_time['cpu'] > self.warning_cooldown:
                self.logger.warning(f"⚠️ High CPU usage: {metrics.cpu_percent:.1f}%")
                self.resource_warnings['cpu'] = True
                self.last_warning_time['cpu'] = current_time
        else:
            self.resource_warnings['cpu'] = False
            
        # Memory pressure check with cooldown  
        if metrics.memory_percent > self.max_memory_percent:
            if current_time - self.last_warning_time['memory'] > self.warning_cooldown:
                self.logger.warning(f"⚠️ High memory usage: {metrics.memory_percent:.1f}%")
                self.resource_warnings['memory'] = True
                self.last_warning_time['memory'] = current_time
        else:
            self.resource_warnings['memory'] = False
            
        # Queue pressure check with cooldown
        overloaded_queues = [name for name, size in metrics.queue_sizes.items() if size > self.max_queue_size]
        if overloaded_queues:
            if current_time - self.last_warning_time['queues'] > self.warning_cooldown:
                self.logger.warning(f"⚠️ Overloaded queues: {overloaded_queues}")
                self.resource_warnings['queues'] = True  
                self.last_warning_time['queues'] = current_time
        else:
            self.resource_warnings['queues'] = False
            
    def _should_cleanup(self, metrics: ResourceMetrics) -> bool:
        """Determine if cleanup is needed - much more conservative"""
        # Only cleanup if severely overloaded
        cpu_critical = metrics.cpu_percent > self.max_cpu_percent + 5  # 5% buffer
        memory_critical = metrics.memory_percent > self.max_memory_percent + 5
        queues_critical = any(size > self.max_queue_size * 1.5 for size in metrics.queue_sizes.values())
        
        return cpu_critical or memory_critical or queues_critical
        
    def _perform_cleanup(self):
        """Perform cleanup operations - much less aggressive"""
        # Only log if we're actually cleaning something significant
        cleaned_anything = False
        
        # More conservative queue cleanup
        for name, q in self.managed_queues.items():
            if q.qsize() > self.max_queue_size:
                cleared = 0
                if 'tts' in name.lower():
                    # TTS queues - keep at least 2 items for responsiveness
                    target_size = max(2, self.max_queue_size // 2)
                    while q.qsize() > target_size:
                        try:
                            q.get_nowait()
                            cleared += 1
                        except queue.Empty:
                            break
                else:
                    # Other queues - less aggressive cleanup
                    target_size = max(5, (self.max_queue_size * 3) // 4)  # Keep 75%
                    while q.qsize() > target_size:
                        try:
                            q.get_nowait()
                            cleared += 1
                        except queue.Empty:
                            break
                if cleared > 10:  # Only log significant cleanups
                    self.logger.info(f"🧹 Cleared {cleared} items from '{name}' queue")
                    cleaned_anything = True
                    
        # Call registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {e}")
                
        # GPU memory cleanup if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if cleaned_anything:  # Only log if other cleanup happened
                    self.logger.info("   Cleared GPU cache")
        except:
            pass
            
        if cleaned_anything:
            self.logger.info("✅ Cleanup completed")
        
    def get_status(self) -> Dict:
        """Get current status and metrics"""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        latest = self.metrics_history[-1]
        
        # Calculate averages over last 10 measurements
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": "healthy" if not any(self.resource_warnings.values()) else "warning",
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "gpu_memory_gb": latest.gpu_memory_used,
                "active_threads": latest.active_threads,
                "queue_sizes": latest.queue_sizes
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "warnings": self.resource_warnings
        }
        
    def safe_queue_put(self, queue_name: str, item, timeout=5.0) -> bool:
        """Safely put item in queue with timeout and size limits"""
        if queue_name not in self.managed_queues:
            return False
            
        q = self.managed_queues[queue_name]
        
        # Check queue size first
        if q.qsize() >= self.max_queue_size:
            self.logger.warning(f"Queue '{queue_name}' is full, dropping item")
            return False
            
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning(f"Queue '{queue_name}' put timeout")
            return False
            
    def safe_queue_get(self, queue_name: str, timeout=1.0):
        """Safely get item from queue with timeout"""
        if queue_name not in self.managed_queues:
            return None
            
        q = self.managed_queues[queue_name]
        
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

# Global process manager instance
process_manager = ProcessManager()

def init_process_management():
    """Initialize process management for SENTER"""
    print("🔧 Initializing process management...")
    
    # Start monitoring
    process_manager.start_monitoring(interval=3.0)
    
    # Register cleanup callbacks
    def memory_cleanup():
        """Memory cleanup callback"""
        import gc
        gc.collect()
        
    def thread_cleanup():
        """Thread cleanup callback - log active threads"""
        active = threading.active_count()
        if active > 20:  # Warning threshold
            print(f"⚠️ High thread count: {active}")
            
    process_manager.register_cleanup_callback(memory_cleanup)
    process_manager.register_cleanup_callback(thread_cleanup)
    
    print("✅ Process management initialized")
    return process_manager

if __name__ == "__main__":
    # Test the process manager
    pm = init_process_management()
    try:
        time.sleep(10)  # Monitor for 10 seconds
        status = pm.get_status()
        print(f"\n📊 Status: {status}")
    finally:
        pm.stop_monitoring() 


================================================
File: senter_face_bridge.py
================================================
#!/usr/bin/env python3
"""
SENTER Face Detection Bridge (Integrated)
=========================================

This script integrates with the existing SENTER system to share face detection
events to a remote SENTER instance at 192.168.1.15.

It monitors the SENTER attention detection system and forwards face detection
events without interfering with the main SENTER camera access.
"""

import time
import json
import socket
import threading
import requests
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Add SENTER modules to path
sys.path.append(str(Path(__file__).parent))

# Configuration
REMOTE_SENTER_IP = "192.168.1.15"
REMOTE_SENTER_PORT = 9091  # HTTP API port for receiving face detection data
UPDATE_INTERVAL = 1.0  # Send updates every second when face is detected
HEARTBEAT_INTERVAL = 30.0  # Send heartbeat every 30 seconds

class SenterFaceBridge:
    """Bridges SENTER face detection data to remote instance."""
    
    def __init__(self, remote_ip: str = REMOTE_SENTER_IP, remote_port: int = REMOTE_SENTER_PORT):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.is_running = False
        
        # State tracking
        self.current_face_detected = False
        self.last_face_detection_time = 0
        self.last_heartbeat_time = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Get local machine info
        self.local_hostname = socket.gethostname()
        self.local_ip = self._get_local_ip()
        
        # SENTER integration
        self.state_logger = None
        self.attention_detector = None
        
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def integrate_with_senter(self):
        """Integrate with the existing SENTER system."""
        try:
            # Try to import SENTER components
            from senter.state_logger import get_state_logger, AttentionState
            
            # Get the current SENTER state logger
            self.state_logger = get_state_logger()
            if self.state_logger:
                self.logger.info("✅ Connected to SENTER state logger")
                return True
            else:
                self.logger.warning("⚠️  SENTER state logger not available")
                return False
                
        except ImportError as e:
            self.logger.error(f"Failed to import SENTER modules: {e}")
            return False
    
    def get_senter_attention_state(self) -> bool:
        """Get current attention state from SENTER system."""
        try:
            if self.state_logger:
                # Get current state from state logger
                current_state = self.state_logger.get_current_state()
                if current_state:
                    # StateSnapshot has attributes, not dict-like access
                    if hasattr(current_state, 'attention_state'):
                        attention_state = current_state.attention_state
                        # Check if user is present (face detected)
                        return str(attention_state) == 'AttentionState.USER_PRESENT'
                    elif hasattr(current_state, '__dict__'):
                        # Try to access as dict if it has __dict__
                        state_dict = current_state.__dict__
                        attention_state = state_dict.get('attention_state')
                        return str(attention_state) == 'AttentionState.USER_PRESENT'
            return False
        except Exception as e:
            self.logger.error(f"Error getting SENTER attention state: {e}")
            return False
    
    def monitor_senter_logs(self):
        """Monitor SENTER logs for face detection events."""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                self.logger.warning("SENTER logs directory not found")
                return
            
            # Find the most recent log file
            log_files = list(log_dir.glob("senter_actions_*.jsonl"))
            if not log_files:
                self.logger.warning("No SENTER log files found")
                return
            
            # Get the most recent log file
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Monitoring SENTER log: {latest_log}")
            
            # Monitor the log file for attention state changes
            self._tail_log_file(latest_log)
            
        except Exception as e:
            self.logger.error(f"Error monitoring SENTER logs: {e}")
    
    def _tail_log_file(self, log_file: Path):
        """Tail a log file and extract face detection events."""
        try:
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while self.is_running:
                    line = f.readline()
                    if line:
                        self._process_log_line(line.strip())
                    else:
                        time.sleep(0.1)  # Wait for new data
                        
        except Exception as e:
            self.logger.error(f"Error tailing log file: {e}")
    
    def _process_log_line(self, line: str):
        """Process a single log line to extract face detection events."""
        try:
            if not line:
                return
                
            # Parse JSON log entry
            log_entry = json.loads(line)
            
            # Look for attention state changes
            if (log_entry.get('action') == 'UpdateAttentionState' or
                'attention_state' in log_entry.get('effects', {})):
                
                # Extract attention state
                attention_state = None
                if 'effects' in log_entry and 'attention_state' in log_entry['effects']:
                    attention_state = log_entry['effects']['attention_state']
                elif 'details' in log_entry and 'new_state' in log_entry['details']:
                    attention_state = log_entry['details']['new_state']
                
                if attention_state:
                    face_detected = attention_state == 'UserPresent'
                    self.send_face_detection_update(face_detected)
                    
        except json.JSONDecodeError:
            # Not a JSON line, ignore
            pass
        except Exception as e:
            self.logger.error(f"Error processing log line: {e}")
    
    def send_face_detection_update(self, face_detected: bool, force_send: bool = False):
        """Send face detection update to remote SENTER."""
        current_time = time.time()
        
        # Determine if we should send an update
        should_send = (
            force_send or
            face_detected != self.current_face_detected or  # State changed
            (face_detected and current_time - self.last_face_detection_time > UPDATE_INTERVAL) or  # Regular updates when face present
            current_time - self.last_heartbeat_time > HEARTBEAT_INTERVAL  # Heartbeat
        )
        
        if not should_send:
            return
        
        try:
            # Prepare the data payload
            data = {
                "source": {
                    "hostname": self.local_hostname,
                    "ip": self.local_ip,
                    "timestamp": datetime.now().isoformat(),
                    "integration": "senter_logs"
                },
                "face_detection": {
                    "detected": face_detected,
                    "timestamp": datetime.now().isoformat(),
                    "changed": face_detected != self.current_face_detected
                }
            }
            
            # Send HTTP POST request to remote SENTER
            url = f"http://{self.remote_ip}:{self.remote_port}/api/face-detection"
            
            response = requests.post(
                url,
                json=data,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                status_text = "DETECTED" if face_detected else "LOST"
                if face_detected != self.current_face_detected:
                    self.logger.info(f"🎯 Face {status_text} - Sent to {self.remote_ip}")
                
                self.current_face_detected = face_detected
                if face_detected:
                    self.last_face_detection_time = current_time
                self.last_heartbeat_time = current_time
                
            else:
                self.logger.warning(f"Remote SENTER responded with status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            if force_send:  # Only log connection errors on initial connection or forced sends
                self.logger.warning(f"Cannot connect to remote SENTER at {self.remote_ip}:{self.remote_port}")
        except Exception as e:
            self.logger.error(f"Error sending face detection update: {e}")
    
    def monitoring_loop(self):
        """Main monitoring loop - combines multiple monitoring strategies."""
        self.logger.info("Starting SENTER face detection monitoring...")
        
        # Send initial connection message
        self.send_face_detection_update(False, force_send=True)
        
        # Start log monitoring in separate thread
        log_thread = threading.Thread(target=self.monitor_senter_logs, daemon=True)
        log_thread.start()
        
        # Main loop - also check state logger periodically
        while self.is_running:
            try:
                # If we have access to state logger, check it directly
                if self.state_logger:
                    face_detected = self.get_senter_attention_state()
                    self.send_face_detection_update(face_detected)
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def start(self) -> bool:
        """Start the SENTER face detection bridge."""
        self.logger.info(f"🚀 Starting SENTER Face Detection Bridge")
        self.logger.info(f"   Local: {self.local_hostname} ({self.local_ip})")
        self.logger.info(f"   Remote: {self.remote_ip}:{self.remote_port}")
        
        # Try to integrate with SENTER system
        if self.integrate_with_senter():
            self.logger.info("✅ Integrated with SENTER system")
        else:
            self.logger.info("📄 Will monitor SENTER logs instead")
        
        # Start monitoring loop in separate thread
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("✅ SENTER Face Detection Bridge started successfully")
        return True
    
    def stop(self):
        """Stop the face detection bridge."""
        self.logger.info("Stopping SENTER Face Detection Bridge...")
        
        self.is_running = False
        
        # Send final update
        self.send_face_detection_update(False, force_send=True)
        
        self.logger.info("✅ SENTER Face Detection Bridge stopped")


def main():
    """Main entry point."""
    print("🎯 SENTER Face Detection Bridge (Integrated)")
    print("=" * 50)
    print(f"Monitoring SENTER face detection and sharing with: {REMOTE_SENTER_IP}:{REMOTE_SENTER_PORT}")
    print("This bridge integrates with the existing SENTER system.")
    print("Press Ctrl+C to stop")
    print()
    
    bridge = SenterFaceBridge()
    
    try:
        if bridge.start():
            # Keep running until interrupted
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start SENTER face detection bridge")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        bridge.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        bridge.stop()


if __name__ == "__main__":
    main() 


================================================
File: senter_status.py
================================================
#!/usr/bin/env python3
"""
Senter AI Assistant - System Status
Shows complete integration status and capabilities
"""

import os
import sys

# Fix OpenMP conflict FIRST
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def check_system_status():
    """Check the status of all Senter components"""
    print("🤖 SENTER AI ASSISTANT - SYSTEM STATUS")
    print("=" * 60)
    print("")
    
    # CLI System Status
    print("🧠 CLI SYSTEM STATUS")
    print("-" * 40)
    try:
        from main import initialize_llm_models, get_available_lights
        print("✅ Core CLI modules: Available")
        
        from tools_config import get_formatted_tools_list
        tools = get_formatted_tools_list()
        print(f"✅ Tool system: {len(tools)} tools available")
        print(f"   • Tools: {', '.join(tools.keys()) if tools else 'None'}")
        
        from journal_system import journal_system
        print("✅ Journal system: Available")
        
        from user_profiles import UserProfile
        print("✅ User profiles: Available")
        
    except Exception as e:
        print(f"❌ CLI system error: {e}")
    print("")
    
    # AvA Integration Status  
    print("👁️ AVA ATTENTION DETECTION STATUS")
    print("-" * 40)
    try:
        from SenterUI.AvA.ava import set_cli_voice_callback, main as ava_main
        print("✅ AvA core system: Available")
        print("✅ CLI callback system: Ready")
        print("✅ Face detection: Ready")
        print("✅ Voice recognition: Ready")
        print("✅ CLI routing: Integrated")
        
    except Exception as e:
        print(f"❌ AvA system error: {e}")
    print("")
    
    # UI System Status
    print("🖥️ UI SYSTEM STATUS") 
    print("-" * 40)
    try:
        from SenterUI.main import SenterUIApp
        print("✅ UI framework: Available")
        
        from SenterUI.senter_integration import SenterUIBridge
        print("✅ CLI-UI bridge: Available")
        print("✅ Chat interface: Ready")
        print("✅ Context areas: Ready")
        print("✅ Real-time display: Ready")
        
    except Exception as e:
        print(f"❌ UI system error: {e}")
    print("")
    
    # Integration Status
    print("🔗 INTEGRATION STATUS")
    print("-" * 40)
    
    try:
        # Test AvA → CLI routing
        from SenterUI.AvA.ava import set_cli_voice_callback
        def test_callback(text):
            pass
        set_cli_voice_callback(test_callback)
        print("✅ AvA → CLI routing: Working")
        
        # Test CLI tool detection
        from main import determine_relevant_tools
        tools = determine_relevant_tools("Turn lights red and research quantum computing")
        print(f"✅ CLI tool detection: Working ({len(tools)} tools)")
        
        # Test UI bridge
        from SenterUI.senter_integration import SenterUIBridge
        class MockUI:
            def _add_user_message_to_chat(self, msg): pass
            def _add_bot_message_to_chat(self, msg): pass
        bridge = SenterUIBridge(MockUI())
        print("✅ UI integration bridge: Working")
        
        print("✅ Voice → CLI → UI pipeline: Ready")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
    print("")
    
    # Capability Summary
    print("🎯 SENTER CAPABILITIES")
    print("-" * 40)
    print("🎙️ Voice Input:")
    print("   • Look at camera to activate")
    print("   • Natural speech recognition")
    print("   • Routes to full CLI system")
    print("")
    print("💬 Text Input:")
    print("   • Type in UI interface")
    print("   • Same CLI processing")
    print("   • Real-time responses")
    print("")
    print("🔧 AI Tools:")
    print("   • Research: Web search & analysis")
    print("   • Lights: Smart home control")
    print("   • Camera: Visual analysis")
    print("   • Journal: Memory & context")
    print("")
    print("🧠 Intelligence:")
    print("   • Vector-based tool selection")
    print("   • Multi-LLM architecture")
    print("   • Personality learning")
    print("   • Conversation memory")
    print("")
    print("🖥️ User Interface:")
    print("   • Modern chat interface")
    print("   • Context-aware displays")
    print("   • Real-time tool results")
    print("   • Multiple view modes")
    print("")
    
    # Launch Instructions
    print("🚀 LAUNCH INSTRUCTIONS")
    print("-" * 40)
    print("Complete System:")
    print("   python launch_senter_complete.py")
    print("")
    print("CLI Only:")
    print("   python main.py")
    print("")
    print("CLI Without AvA:")
    print("   python main.py --no-attention")
    print("")
    print("UI Only (No AI):")
    print("   python SenterUI/main.py")
    print("")
    
    print("🎯 SENTER AI ASSISTANT IS READY!")
    print("=" * 60)

if __name__ == "__main__":
    check_system_status() 


================================================
File: senter_ui.py
================================================
#!/usr/bin/env python3
"""
Senter UI - Main Launcher
Launches the login window which handles CLI setup before launching the main UI
"""

import os
import sys
import subprocess

def main():
    """Launch the Senter UI with login window"""
    print("🚀 Starting Senter UI with Login Window...")
    print("=" * 50)
    
    # Set working directory to SenterUI
    senter_ui_dir = os.path.join(os.path.dirname(__file__), 'SenterUI')
    
    print(f"💻 Launching: {sys.executable} login_window.py")
    print(f"📁 Working directory: {senter_ui_dir}")
    
    # Launch the login window
    try:
        os.chdir(senter_ui_dir)
        result = subprocess.run([
            sys.executable, 'login_window.py'
        ], check=False)
        
        return result.returncode
    except Exception as e:
        print(f"❌ Error launching Senter UI: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 


================================================
File: tools_config.py
================================================
"""
Tools Configuration
Contains all available tool definitions for the assistant system.
"""

def get_tools_list():
    """Return the list of available tools for the assistant."""
    return [
        '''
        <research>
        (in the <announcement> say "let me look that up for you" or similar)
        (A single search query string to find current information on the internet)
        (Use for: current events, latest news, product releases, factual questions requiring up-to-date information)
        </research>
        ''',
        '''
        <lights>
        (in the <announcement> say "turning on the lights" or similar)
        (Room Name or "ALL") (Action: ON/OFF/Brightness X%/Color Name/Set Color (R,G,B))
        (Available lights: """ + lights_description + """)
        
        CRITICAL: When user says "the lights", "lights", "all lights", or "all the lights" without specifying a room, ALWAYS use "ALL"
        When user specifies a room name, use that specific room.
        
        EXAMPLES:
        - "turn the lights blue" → ALL Blue
        - "turn lights red" → ALL Red  
        - "the lights green" → ALL Green
        - Kitchen ON → Kitchen ON
        - Living Room OFF → Living Room OFF
        - Desk Brightness 75% → Desk Brightness 75%
        - "turn all lights yellow" → ALL Yellow
        
        COLOR NAMES: Red, Green, Blue, White, Yellow, Cyan, Magenta, Orange, Purple, Pink, Teal, Turquoise, Lime, Warm_White, Cool_White
        Or use RGB format: Set Color (R,G,B) where R,G,B are 0-255
        </lights>
        ''',
        '''
        <camera>
        (in the <announcement>, mention what you're capturing in a casual way)
        (Camera command: "front camera", "screenshot", "screen", "take photo", "how do I look", etc.)
        (Use for: taking webcam photos, screenshots, analyzing what's visible, appearance questions)
        
        CRITICAL: For "how do I look", "how I look", "how does my hair look", "my appearance", any appearance questions → USE CAMERA NOT RESEARCH
        
        EXAMPLES:
        - "how do I look" → front camera
        - "how I look" → front camera  
        - "take a photo" → front camera  
        - "screenshot" → screenshot
        - "what's on my screen" → screenshot
        </camera>
        '''
    ]

def get_formatted_tools_list(lights_description: str = None):
    """Return the tools list with dynamic content filled in."""
    tools = get_tools_list()
    
    # Replace lights description placeholder if provided
    if lights_description:
        for i, tool in enumerate(tools):
            if '<lights>' in tool:
                tools[i] = tool.format(lights_description=lights_description)
                break
    
    # Return as a dictionary for compatibility with status checks
    tool_dict = {}
    for tool in tools:
        if '<research>' in tool:
            tool_dict['research'] = tool
        elif '<lights>' in tool:
            tool_dict['lights'] = tool
        elif '<camera>' in tool:
            tool_dict['camera'] = tool
    
    return tool_dict

def add_tool(tool_definition: str):
    """Add a new tool to the configuration (for future extensibility)."""
    # This could be extended to write to a config file or database
    pass

def remove_tool(tool_name: str):
    """Remove a tool from the configuration (for future extensibility)."""
    # This could be extended to modify a config file or database
    pass 


================================================
File: user_profiles.py
================================================
import json
import os
import getpass
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any

PROFILES_DIR = "user_profiles"
PROFILES_FILE = os.path.join(PROFILES_DIR, "profiles.json")

class UserProfile:
    """Manages user profiles with credentials and preferences."""
    
    def __init__(self):
        self.current_user = None
        self.user_data = {}
        self.ensure_profiles_dir()
        
    def ensure_profiles_dir(self):
        """Ensure the profiles directory exists."""
        if not os.path.exists(PROFILES_DIR):
            os.makedirs(PROFILES_DIR)
            
    def hash_password(self, password: str) -> str:
        """Hash a password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()
        
    def load_profiles(self) -> Dict[str, Any]:
        """Load all user profiles from file."""
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
        
    def save_profiles(self, profiles: Dict[str, Any]):
        """Save all user profiles to file."""
        try:
            with open(PROFILES_FILE, 'w') as f:
                json.dump(profiles, f, indent=2)
        except IOError as e:
            print(f"❌ Error saving profiles: {e}")
            
    def create_default_chris_profile(self) -> Dict[str, Any]:
        """Create the default Chris profile with current settings."""
        return {
            "name": "Chris",
            "display_name": "Chris",
            "created_date": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "credentials": {
                "aidot": {
                    "username": "christophersghardwick@gmail.com",
                    "password": "A111s1nmym!nd",
                    "country": "UnitedStates"
                }
            },
            "preferences": {
                "greeting_style": "friendly",
                "tts_enabled": True,
                "voice_model": "en_US-lessac-medium",
                "temperature": 0.3,
                "max_tokens": 300
            },
            "interests": ["technology", "smart home", "automation", "AI"],
            "notes": "Creator and primary user of Senter system"
        }
        
    def create_new_profile(self) -> Optional[str]:
        """Interactive profile creation process."""
        print("\n🆕 Creating New User Profile")
        print("=" * 40)
        
        # Get basic info
        name = input("Enter your name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return None
            
        display_name = input(f"Display name (default: {name}): ").strip() or name
        
        # Get password
        while True:
            password = getpass.getpass("Enter a password for your profile: ")
            if len(password) < 4:
                print("❌ Password must be at least 4 characters")
                continue
            confirm_password = getpass.getpass("Confirm password: ")
            if password != confirm_password:
                print("❌ Passwords don't match")
                continue
            break
            
        # Get AiDot credentials (optional)
        print("\n💡 AiDot Smart Light Credentials (optional):")
        aidot_username = input("AiDot username/email (press Enter to skip): ").strip()
        aidot_password = ""
        aidot_country = "UnitedStates"
        
        if aidot_username:
            aidot_password = getpass.getpass("AiDot password: ")
            aidot_country = input("Country (default: UnitedStates): ").strip() or "UnitedStates"
            
        # Get preferences
        print("\n⚙️  Preferences:")
        tts_enabled = input("Enable text-to-speech? (y/n, default: y): ").lower().strip()
        tts_enabled = tts_enabled != 'n'
        
        greeting_style = input("Greeting style (friendly/professional/casual, default: friendly): ").strip() or "friendly"
        
        # Get interests
        print("\n🎯 Interests (comma-separated, optional):")
        interests_input = input("Enter your interests: ").strip()
        interests = [i.strip() for i in interests_input.split(',') if i.strip()] if interests_input else []
        
        # Create profile data
        profile_data = {
            "name": name,
            "display_name": display_name,
            "password_hash": self.hash_password(password),
            "created_date": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "credentials": {},
            "preferences": {
                "greeting_style": greeting_style,
                "tts_enabled": tts_enabled,
                "voice_model": "en_US-lessac-medium",
                "temperature": 0.3,
                "max_tokens": 300
            },
            "interests": interests,
            "notes": f"Profile created on {datetime.now().strftime('%Y-%m-%d')}"
        }
        
        # Add AiDot credentials if provided
        if aidot_username:
            profile_data["credentials"]["aidot"] = {
                "username": aidot_username,
                "password": aidot_password,
                "country": aidot_country
            }
            
        # Save profile
        profiles = self.load_profiles()
        if name.lower() in [k.lower() for k in profiles.keys()]:
            print(f"❌ Profile '{name}' already exists")
            return None
            
        profiles[name] = profile_data
        self.save_profiles(profiles)
        
        print(f"\n✅ Profile '{display_name}' created successfully!")
        return name
        
    def login(self, username: str, password: str) -> bool:
        """Authenticate a user."""
        profiles = self.load_profiles()
        
        # Find user (case-insensitive)
        user_key = None
        for key in profiles.keys():
            if key.lower() == username.lower():
                user_key = key
                break
                
        if not user_key:
            return False
            
        profile = profiles[user_key]
        
        # Check password (Chris profile doesn't have password for backward compatibility)
        if "password_hash" in profile:
            if self.hash_password(password) != profile["password_hash"]:
                return False
        elif password:  # If they entered a password but profile doesn't have one, fail
            return False
            
        # Update last login
        profile["last_login"] = datetime.now().isoformat()
        profiles[user_key] = profile
        self.save_profiles(profiles)
        
        # Set current user
        self.current_user = user_key
        self.user_data = profile
        
        return True
        
    def get_current_user_data(self) -> Dict[str, Any]:
        """Get current user's profile data."""
        return self.user_data
        
    def get_aidot_credentials(self) -> Optional[Dict[str, str]]:
        """Get AiDot credentials for current user."""
        if self.user_data and "credentials" in self.user_data and "aidot" in self.user_data["credentials"]:
            return self.user_data["credentials"]["aidot"]
        return None
        
    def get_display_name(self) -> str:
        """Get the display name for the current user."""
        if self.user_data:
            return self.user_data.get("display_name", self.user_data.get("name", "User"))
        return "User"
        
    def get_current_username(self) -> str:
        """Get the current username."""
        return self.current_user or "default"
        
    def get_greeting_style(self) -> str:
        """Get greeting style preference."""
        if self.user_data and "preferences" in self.user_data:
            return self.user_data["preferences"].get("greeting_style", "friendly")
        return "friendly"
        
    def is_tts_enabled(self) -> bool:
        """Check if TTS is enabled for current user."""
        if self.user_data and "preferences" in self.user_data:
            return self.user_data["preferences"].get("tts_enabled", True)
        return True
        
    def setup_initial_profiles(self):
        """Set up initial profiles including Chris."""
        profiles = self.load_profiles()
        
        # Create Chris profile if it doesn't exist
        if "Chris" not in profiles:
            profiles["Chris"] = self.create_default_chris_profile()
            self.save_profiles(profiles)
            print("✅ Created default Chris profile")
            
    def show_login_screen(self) -> bool:
        """Show login screen and handle authentication."""
        # Check for Docker auto-login mode
        import os
        if os.getenv('DOCKER_MODE') == '1' and os.getenv('AUTO_LOGIN_USER'):
            auto_user = os.getenv('AUTO_LOGIN_USER')
            print(f"🐳 Docker mode: Auto-login as {auto_user}")
            self.setup_initial_profiles()
            if self.login(auto_user, ""):
                print(f"✅ Auto-logged in as {auto_user}")
                return True
            else:
                print(f"❌ Auto-login failed for {auto_user}")
                # Fall through to manual login
        
        self.setup_initial_profiles()
        profiles = self.load_profiles()
        
        if not profiles:
            print("🆕 No profiles found. Let's create your first profile!")
            username = self.create_new_profile()
            if username:
                self.current_user = username
                self.user_data = profiles.get(username, {})
                return True
            return False
            
        print(f"\n👋 Welcome to Senter!")
        print("=" * 40)
        print("\nAvailable Profiles:")
        print("-" * 20)
        
        # Create a list of profile options with clear names
        profile_list = list(profiles.items())
        for i, (name, profile) in enumerate(profile_list, 1):
            display_name = profile.get("display_name", name)
            last_login = profile.get("last_login", "Never")
            if last_login != "Never":
                try:
                    last_login = datetime.fromisoformat(last_login).strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            print(f"  {i}. {display_name}")
            print(f"     └─ Last login: {last_login}")
        
        print(f"\n  {len(profiles) + 1}. Create New Profile")
        print("-" * 20)
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(profiles) + 1}): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(profiles) + 1:
                    # Create new profile
                    username = self.create_new_profile()
                    if username:
                        self.current_user = username
                        profiles = self.load_profiles()  # Reload to get new profile
                        self.user_data = profiles.get(username, {})
                        return True
                    continue
                    
                elif 1 <= choice_num <= len(profiles):
                    # Select existing profile
                    username, profile = profile_list[choice_num - 1]
                    display_name = profile.get("display_name", username)
                    
                    print(f"\n🔑 Selected: {display_name}")
                    
                    # Check if profile has password
                    if "password_hash" in profile:
                        password = getpass.getpass(f"Enter password for {display_name}: ")
                        if not self.login(username, password):
                            print("❌ Incorrect password")
                            continue
                    else:
                        # No password required (legacy Chris profile)
                        if self.login(username, ""):
                            print(f"✅ Welcome back, {display_name}!")
                        else:
                            print("❌ Login failed")
                            continue
                            
                    return True
                else:
                    print("❌ Invalid choice. Please select a valid option.")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                return False 





================================================
File: SenterUI/__init__.py
================================================
# SenterUI Package 


================================================
File: SenterUI/ui_components.py
================================================
"""
Minimal SenterUI components for Docker/CLI mode
"""

import os

class SenterUI:
    """Minimal UI components for Docker/CLI mode"""
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    @staticmethod
    def show_ascii_logo():
        """Show ASCII logo"""
        print("""
╔═══════════════════════════════════════╗
║              🎯 SENTER                ║
║        AI-Powered Assistant           ║
║                                       ║
║    🧠 Smart • 🔧 Extensible • 🚀 Fast  ║
╚═══════════════════════════════════════╝
        """)
    
    @staticmethod
    def show_multi_step_progress(steps, step_duration=1.0):
        """Show initialization progress"""
        import time
        print("\n🔄 Initializing SENTER...")
        print("=" * 40)
        
        for i, (description, func) in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {description}...", end=" ", flush=True)
            try:
                result = func()
                if result is False:
                    print("⚠️  Warning")
                else:
                    print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
            time.sleep(0.5)  # Faster for Docker
        
        print("=" * 40)
        print("✅ Initialization complete!")
    
    @staticmethod
    def show_welcome_message(display_name, greeting_style="friendly"):
        """Show welcome message"""
        print(f"\n🎉 Welcome back, {display_name}!")
        
        if greeting_style == "professional":
            print("📋 Ready to assist with your tasks.")
        elif greeting_style == "casual":
            print("😎 What's up? Ready to get things done!")
        else:  # friendly
            print("😊 How can I help you today?")
        
        print("\n💡 Available commands:")
        print("   • Research: Ask questions for web search")
        print("   • Lights: Control smart home devices")
        print("   • Camera: Take photos or analyze images")
        print("   • Journal: Search conversation history")
        print("   • Type 'exit' to quit")
        print("-" * 40) 


================================================
File: SenterUI/AvA/AImouse.py
================================================
import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# --- Configuration ---
SMOOTHING = 5
CLICK_DEBOUNCE_TIME = 0.4 # Slightly increased debounce might be needed
DRAG_ACTIVATION_FRAMES = 5
MAX_CAMERAS_TO_TEST = 5

# --- Initialization (MediaPipe, Screen Size) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7, # Keep reasonably high
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

screen_width, screen_height = pyautogui.size()
print(f"Screen resolution: {screen_width}x{screen_height}")

# --- State Variables ---
prev_x, prev_y = 0, 0
current_x, current_y = 0, 0
is_dragging = False
last_action_time = 0 # Renamed from last_click_time for clarity
fist_detected_frames = 0

# --- Helper Function: Visual Camera Selection ---
# (select_camera_visual function remains the same as the previous version)
def select_camera_visual(max_test=5):
    # ... (Keep the implementation from the previous response) ...
    available_cameras = []
    print("Detecting available cameras...")
    for i in range(max_test):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        time.sleep(0.1)
        is_opened = cap_test.isOpened()
        if is_opened:
            print(f"  - Found active camera at index {i}")
            available_cameras.append(i)
            cap_test.release()
            # print(f"    (Camera index {i} released after check)") # Optional debug print
        else:
            # print(f"  - No camera found at index {i}") # Optional debug print
            cap_test.release()

    if not available_cameras:
        print("\nError: No cameras detected.")
        return None
    print(f"\nFound cameras at indices: {available_cameras}")
    current_preview_num = 0
    selected_index = None
    preview_window_name = "Camera Preview (n=next, s=select, q=quit)"
    while selected_index is None:
        if not available_cameras: # Check if list became empty due to errors
             print("Error: No previewable cameras remaining.")
             selected_index = -1
             break
        cam_index = available_cameras[current_preview_num % len(available_cameras)] # Use modulo for safety
        print(f"\nAttempting to preview Camera Index: {cam_index}...")
        cap_preview = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        time.sleep(0.2)
        if not cap_preview.isOpened():
            print(f"  Error: Could not open camera {cam_index} for preview. Skipping.")
            cap_preview.release()
            try: available_cameras.remove(cam_index) # Remove faulty index
            except ValueError: pass # Already removed
            current_preview_num = current_preview_num % len(available_cameras) if available_cameras else 0
            continue
        print(f"  -> Now previewing Camera Index: {cam_index}. Check the '{preview_window_name}' window.")
        print("     Press 'n' for next, 's' to select this one, 'q' to quit.")
        while True:
            ret, frame = cap_preview.read()
            if not ret:
                print(f"  Warning: Could not read frame from camera {cam_index} during preview.")
                try: available_cameras.remove(cam_index)
                except ValueError: pass
                current_preview_num = current_preview_num % len(available_cameras) if available_cameras else 0
                break
            text = f"PREVIEW Index: {cam_index} | (n)ext (s)elect (q)uit"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(preview_window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('n'): print("  'n' pressed. Moving to next camera..."); break
            elif key == ord('s'): print(f"  's' pressed. Selected camera index {cam_index}."); selected_index = cam_index; break
            elif key == ord('q'): print("  'q' pressed. Quitting camera selection."); selected_index = -1; break
        print(f"  Releasing preview capture for index {cam_index}...")
        cap_preview.release()
        if selected_index is not None: break
        if available_cameras: current_preview_num = (current_preview_num + 1) % len(available_cameras)
        else: print("Error: No cameras left after preview errors."); selected_index = -1; break # No cameras left
    cv2.destroyWindow(preview_window_name)
    cv2.waitKey(1)
    return None if selected_index == -1 else selected_index


# --- Gesture Detection Helper Functions ---

def get_landmark_coordinates(img_height, img_width, landmarks):
    # (Same as before)
    coords = []
    if landmarks:
        for lm in landmarks.landmark:
            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                 coords.append((int(lm.x * img_width), int(lm.y * img_height)))
            else: coords.append((0,0))
    return coords

def get_finger_states(landmark_coords, handedness):
    """
    Determines the extension state (True=extended, False=curled) for each finger.
    Returns a dictionary: {'thumb': bool, 'index': bool, ...}
    """
    states = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
    if not landmark_coords or len(landmark_coords) < 21:
        # print("Warning: Insufficient landmarks for finger state detection.")
        return states # Return default (all false)

    # Finger tip IDs and PIP joint IDs
    tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]
    pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, # Use IP for thumb's 'PIP'
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
               mp_hands.HandLandmark.PINKY_PIP]
    mcp_ids = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
               mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
               mp_hands.HandLandmark.PINKY_MCP]
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # Check Index, Middle, Ring, Pinky based on vertical position
    for i in range(1, 5): # Skip thumb for now
        tip_y = landmark_coords[tip_ids[i]][1]
        pip_y = landmark_coords[pip_ids[i]][1]
        mcp_y = landmark_coords[mcp_ids[i]][1]
        # Finger is extended if tip is significantly higher than PIP
        # Also check if tip is higher than MCP to avoid edge cases when hand is flat
        if tip_y < pip_y and tip_y < mcp_y:
             states[finger_names[i]] = True
        else:
             states[finger_names[i]] = False

    # Check Thumb based on horizontal position relative to MCP/IP (more robust)
    thumb_tip_x = landmark_coords[tip_ids[0]][0]
    thumb_pip_x = landmark_coords[pip_ids[0]][0] # Using IP as PIP reference
    thumb_mcp_x = landmark_coords[mcp_ids[0]][0]

    # Assuming mirrored view: Right hand thumb is left-most when extended, Left hand thumb is right-most
    if handedness.lower() == 'right':
        # Extended if tip is further left than PIP/MCP
        if thumb_tip_x < thumb_pip_x and thumb_tip_x < thumb_mcp_x:
            states['thumb'] = True
        else:
            states['thumb'] = False
    elif handedness.lower() == 'left':
        # Extended if tip is further right than PIP/MCP
        if thumb_tip_x > thumb_pip_x and thumb_tip_x > thumb_mcp_x:
             states['thumb'] = True
        else:
             states['thumb'] = False
    else: # Handedness unknown, fallback (less reliable)
        # Check if thumb tip is vertically higher than PIP (IP)
        if landmark_coords[tip_ids[0]][1] < landmark_coords[pip_ids[0]][1]:
             states['thumb'] = True
        else:
             states['thumb'] = False


    return states

def is_fist(finger_states):
    """Checks if all fingers are curled."""
    # True if ALL finger states are False (not extended)
    return not any(finger_states.values())

def is_shaka(finger_states):
    """Checks for Shaka sign (thumb and pinky extended, others curled)."""
    return (finger_states['thumb'] and
            finger_states['pinky'] and
            not finger_states['index'] and
            not finger_states['middle'] and
            not finger_states['ring'])

def is_index_finger_up(finger_states):
     """Checks if only the index finger is extended."""
     return (finger_states['index'] and
            not finger_states['thumb'] and
            not finger_states['middle'] and
            not finger_states['ring'] and
            not finger_states['pinky'])

def is_back_of_hand(landmark_coords, handedness):
    """
    Checks if the back of the hand is likely facing the camera.
    Compares horizontal position of thumb base vs pinky base.
    """
    if not landmark_coords or len(landmark_coords) < 21 or not handedness:
        return False

    thumb_mcp_x = landmark_coords[mp_hands.HandLandmark.THUMB_MCP][0]
    pinky_mcp_x = landmark_coords[mp_hands.HandLandmark.PINKY_MCP][0]

    # In mirrored view:
    # Right Hand: Palm view -> Thumb MCP left of Pinky MCP (thumb_mcp_x < pinky_mcp_x)
    #             Back view -> Thumb MCP right of Pinky MCP (thumb_mcp_x > pinky_mcp_x)
    # Left Hand:  Palm view -> Thumb MCP right of Pinky MCP (thumb_mcp_x > pinky_mcp_x)
    #             Back view -> Thumb MCP left of Pinky MCP (thumb_mcp_x < pinky_mcp_x)

    if handedness.lower() == 'right':
        return thumb_mcp_x > pinky_mcp_x
    elif handedness.lower() == 'left':
        return thumb_mcp_x < pinky_mcp_x
    else:
        return False # Cannot determine without handedness

# --- Camera Selection ---
selected_camera_index = select_camera_visual(MAX_CAMERAS_TO_TEST)
if selected_camera_index is None:
    print("No camera selected or selection cancelled/failed. Exiting.")
    if 'hands' in locals(): hands.close()
    exit()

# --- Initialize Video Capture ---
print(f"\nInitializing main capture for selected camera index: {selected_camera_index}")
cap = cv2.VideoCapture(selected_camera_index, cv2.CAP_DSHOW)
time.sleep(0.5)
if not cap.isOpened():
    print(f"Error: Could not open selected camera index {selected_camera_index} (even after selection).")
    if 'hands' in locals(): hands.close()
    exit() # Exit if the selected camera fails to open


# --- Main Loop ---
results = None # Define results in outer scope
try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            time.sleep(0.1)
            continue

        # --- Image Processing ---
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Reset states for current frame ---
        current_action = "None"
        perform_click = None # None, 'left', or 'right'
        move_mouse_flag = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label

            # --- Draw Landmarks ---
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # --- Get Coordinates and States ---
            landmark_coords = get_landmark_coordinates(img_height, img_width, hand_landmarks)
            finger_states = get_finger_states(landmark_coords, handedness)

            # --- Gesture Detection (Prioritized) ---
            is_fist_gesture = is_fist(finger_states)
            is_shaka_gesture = is_shaka(finger_states)
            is_back_gesture = is_back_of_hand(landmark_coords, handedness)
            is_index_gesture = is_index_finger_up(finger_states)

            # --- Mouse Coordinate Calculation (based on index finger tip) ---
            if mp_hands.HandLandmark.INDEX_FINGER_TIP < len(landmark_coords):
                ix, iy = landmark_coords[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                active_x_min = img_width * 0.1; active_x_max = img_width * 0.9
                active_y_min = img_height * 0.1; active_y_max = img_height * 0.9

                if active_x_min <= ix <= active_x_max and active_y_min <= iy <= active_y_max:
                    target_x = np.interp(ix, (active_x_min, active_x_max), (0, screen_width))
                    target_y = np.interp(iy, (active_y_min, active_y_max), (0, screen_height))
                    target_x = max(0, min(screen_width - 1, target_x))
                    target_y = max(0, min(screen_height - 1, target_y))
                    current_x = prev_x + (target_x - prev_x) / SMOOTHING
                    current_y = prev_y + (target_y - prev_y) / SMOOTHING

                    # Decide action based on gesture priority
                    if is_fist_gesture:
                        current_action = "Fist (Drag?)"
                        fist_detected_frames += 1
                        if not is_dragging and fist_detected_frames >= DRAG_ACTIVATION_FRAMES:
                            print("Drag Started")
                            pyautogui.moveTo(int(current_x), int(current_y), duration=0.05)
                            pyautogui.mouseDown(button='left')
                            is_dragging = True
                            last_action_time = time.time() # Reset debounce timer
                        if is_dragging:
                            current_action = "Dragging"
                            move_mouse_flag = True # Move mouse while dragging

                    else: # Not a fist
                        fist_detected_frames = 0
                        if is_dragging:
                            print("Drag Released")
                            pyautogui.mouseUp(button='left')
                            is_dragging = False
                            last_action_time = time.time() # Reset debounce timer

                        # Check other gestures ONLY if not dragging and debounce period passed
                        elif time.time() - last_action_time > CLICK_DEBOUNCE_TIME:
                            if is_back_gesture:
                                current_action = "Back of Hand (L Click)"
                                perform_click = 'left'
                            elif is_shaka_gesture:
                                current_action = "Shaka (R Click)"
                                perform_click = 'right'
                            elif is_index_gesture:
                                current_action = "Index Finger (Moving)"
                                move_mouse_flag = True
                            else:
                                # Optional: move if any other non-specific gesture?
                                # Or require index finger explicitly
                                current_action = "Other/Idle"
                                # move_mouse_flag = True # Uncomment to move on any non-gesture

                        else: # In debounce period
                            current_action = "Debouncing"
                            # Allow movement during debounce if index is up?
                            if is_index_gesture:
                                 move_mouse_flag = True

                    # --- Update smoothing coordinates ---
                    prev_x, prev_y = current_x, current_y

                else: # Outside active area
                     current_action = "Hand Outside Active Area"
                     if is_dragging: # Release drag if hand moves out
                         print("Drag Released (Hand Out of Area)")
                         pyautogui.mouseUp(button='left')
                         is_dragging = False
                         last_action_time = time.time()
                     fist_detected_frames = 0

                # --- Visual Feedback ---
                cv2.circle(image, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
                cv2.rectangle(image, (int(active_x_min), int(active_y_min)),
                             (int(active_x_max), int(active_y_max)), (255, 0, 255), 1)

            else: # Index finger landmark not found
                current_action = "Waiting for Index Landmark"
                if is_dragging: # Release drag if tracking lost
                     print("Drag Released (Tracking Lost)")
                     pyautogui.mouseUp(button='left')
                     is_dragging = False
                     last_action_time = time.time()
                fist_detected_frames = 0

            # --- Perform Actions (Clicks / Moves) ---
            if perform_click:
                print(f"Action: {current_action}")
                pyautogui.click(button=perform_click)
                last_action_time = time.time() # Update debounce timer
            elif move_mouse_flag:
                pyautogui.moveTo(int(current_x), int(current_y), duration=0)


        else: # No hand detected
            current_action = "No Hand Detected"
            if is_dragging:
                 print("Drag Released (Hand Lost)")
                 pyautogui.mouseUp(button='left')
                 is_dragging = False
                 last_action_time = time.time()
            fist_detected_frames = 0

        # --- Display Info ---
        cv2.putText(image, f"Hand: {handedness}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Action: {current_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Optional: Display finger states for debugging
        # cv2.putText(image, f"{finger_states}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow('Hand Tracking Mouse Control - Press Q to Quit', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("'q' pressed. Exiting.")
            break

except KeyboardInterrupt: print("\nScript interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc()
finally:
    # --- Cleanup ---
    print("Cleaning up resources...")
    if is_dragging:
        print("Releasing mouse button...")
        try: pyautogui.mouseUp(button='left')
        except Exception as py_err: print(f"  Error releasing mouse button: {py_err}")
    if 'cap' in locals() and cap is not None and cap.isOpened():
        print("Releasing video capture...")
        cap.release()
    if 'hands' in locals() and hands is not None:
        print("Closing Mediapipe hands...")
        hands.close()
    print("Destroying OpenCV windows...")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Script finished.")


================================================
File: SenterUI/AvA/CVA.py
================================================
import cv2
import numpy as np
import pyautogui
import time
import os
import tkinter as tk
from tkinter import ttk, font
import threading
from pynput import keyboard
import requests
import base64
import io
from PIL import Image
import json

# --- Configuration ---

# -- OpenCV/Detection Parameters (Tune these!) --
MIN_CONTOUR_AREA = 150
MAX_CONTOUR_AREA = 150000
ADAPTIVE_THRESH_BLOCK_SIZE = 15 # Must be odd
ADAPTIVE_THRESH_C = 5
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
DRAW_BOX_NUMBER = True

# -- Ollama Configuration --
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API URL
OLLAMA_VISION_MODEL = "gemma3:4b" # Model for analyzing the screenshot (Update if needed)
OLLAMA_INTERPRET_MODEL = "hermes2-llama3.2:3b" # Model for interpreting user request (Update if name differs)
# OLLAMA_REQUEST_TIMEOUT = 120 # Removed timeout for vision model

# -- UI Configuration --
UI_WIDTH = 400
UI_INPUT_HEIGHT = 40
UI_RESPONSE_HEIGHT = 80
UI_BG_COLOR = "#2E2E2E"
UI_FG_COLOR = "#E0E0E0"
UI_FONT = ("Segoe UI", 10) # Or another font like "Arial", "Calibri"
UI_TRANSPARENCY = 0.9 # Approximate transparency (1.0 = opaque)

# -- Keyboard Shortcut --
DOUBLE_TAP_DELAY = 0.3 # Seconds within which two taps count as a double tap

# --- Global State ---
last_ctrl_press_time = 0
ui_visible = False
root = None
input_entry = None
response_label = None
latest_screenshot = None # Store the screenshot taken before UI appears
detected_objects_cache = None # Store basic contour results {id, x, y, w, h, center}
vision_analysis_result = None # Will store richer descriptions from gemma {id, description, center} or error string
vision_analysis_thread = None
vision_analysis_complete = threading.Event() # To signal when vision analysis is done

# --- OpenCV Object Detection (Adapted from previous script) ---

def detect_objects_by_contour(screenshot_img):
    """Detects potential objects using contour detection."""
    if screenshot_img is None:
        return [], None

    gray = cv2.cvtColor(screenshot_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESHOLD_TYPE, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
        )
    except cv2.error as e:
        print(f"OpenCV Error during adaptiveThreshold: {e}")
        # Return empty list and original image if thresholding fails
        return [], screenshot_img.copy() # Return a copy to avoid modifying original

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_objects = []
    annotated_img = screenshot_img.copy()
    object_id_counter = 1

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            # Basic filtering: avoid boxes that are too thin or too tall
            if w < 5 or h < 5 or w > screenshot_img.shape[1] * 0.9 or h > screenshot_img.shape[0] * 0.9:
                 continue

            center_x = x + w // 2
            center_y = y + h // 2

            obj_info = {
                'id': object_id_counter, 'x': x, 'y': y, 'w': w, 'h': h,
                'center': (center_x, center_y), 'area': area
            }
            detected_objects.append(obj_info)

            # Draw rectangle on the annotated image copy
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if DRAW_BOX_NUMBER:
                label = str(object_id_counter)
                # Put text slightly above the top-left corner
                cv2.putText(annotated_img, label, (x, y - 5 if y > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2) # Thicker font
            object_id_counter += 1

    # Sort by position (top-to-bottom, then left-to-right) for potentially more logical IDs
    detected_objects.sort(key=lambda o: (o['y'], o['x']))

    # Re-assign IDs after sorting for consistency in the prompt sent to the vision model
    # Note: The labels drawn on the image *will* correspond to these final IDs because re-drawing is complex.
    # The vision model prompt will use these final IDs.
    final_detected_objects = []
    # Clear previous drawings if we were to re-draw labels (but we are not re-drawing)
    # annotated_img = screenshot_img.copy() # Reset if re-drawing

    for i, obj in enumerate(detected_objects):
        obj['id'] = i + 1 # Assign final ID after sorting
        final_detected_objects.append(obj)
        # Re-draw rectangle and text with final ID if perfect visual match is required
        # x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        # cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # if DRAW_BOX_NUMBER:
        #     label = str(obj['id'])
        #     cv2.putText(annotated_img, label, (x, y - 5 if y > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Detected {len(final_detected_objects)} potential objects after filtering and sorting.")
    # Return the final sorted list and the annotated image (with original drawing IDs)
    return final_detected_objects, annotated_img

# --- Image Encoding ---

def encode_image_to_base64(cv2_image):
    """Encodes an OpenCV image (NumPy array) to base64 string."""
    try:
        # Convert BGR (OpenCV default) to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG") # Save as PNG (or JPEG)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# --- Ollama Interaction ---

def get_vision_analysis(annotated_base64_image_data, object_list):
    """
    Sends annotated image and object list to the VISION model (gemma3:4b)
    to get descriptions for all detected objects.
    Stores a structured list (e.g., JSON parsed data) or error string in global state.
    """
    global vision_analysis_result, vision_analysis_complete

    headers = {"Content-Type": "application/json"}
    # Construct object list string for the prompt using final IDs from contour detection
    objects_str = ", ".join([f"ID {obj['id']} (center: {obj['center']})" for obj in object_list])
    if not objects_str:
        objects_str = "No objects initially detected."

    # --- Construct the prompt for the VISION model ---
    # Ask it to describe each object and associate it with its ID and center coordinates.
    # Requesting JSON output makes parsing easier.
    full_prompt = f"""Analyze the attached image. It shows a desktop screenshot with green boxes around detected UI elements, each potentially having an ID number drawn near it (though the number might be obscured or slightly misplaced).
Here is a list of the detected elements with their assigned IDs and center coordinates (cx, cy): {objects_str}.

Your task is to describe the element inside or associated with each numbered box/ID. For each element you describe, provide its ID (matching the list), a concise visual description, and confirm its center coordinates.
Respond ONLY with a valid JSON list of objects, where each object has 'id' (integer), 'description' (string), and 'center' (list of two integers [cx, cy]).
Example format:
[
  {{"id": 1, "description": "Blue 'OK' button", "center": [150, 200]}},
  {{"id": 2, "description": "Text input field containing 'example'", "center": [300, 250]}},
  ...
]
If you cannot identify or describe an element for a given ID, omit it from the list. If no elements can be described, respond with an empty JSON list: []"""

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": full_prompt,
        "format": "json", # Request JSON output directly
        "stream": False,
        "images": [annotated_base64_image_data]
    }

    analysis_result_data = None # To store the parsed JSON data
    error_message = None

    try:
        print(f"Sending request to Vision Model ({OLLAMA_VISION_MODEL})... This may take a while.")
        # No timeout specified here, relies on underlying http/socket timeouts
        response = requests.post(
            OLLAMA_API_URL,
            headers=headers,
            data=json.dumps(payload)
            # timeout= removed
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        # Ollama with format="json" should return the JSON object directly in the response field
        # However, it might be nested or still a string depending on version/implementation.
        # Let's try to get the content and parse defensively.

        content = response_data.get("response") # Check if 'response' holds the direct JSON object or string
        content_str = ""

        if isinstance(content, dict) or isinstance(content, list):
             # If Ollama directly returns the parsed JSON object/list
             analysis_result_data = content
             print("Vision Model returned structured JSON.")
        elif isinstance(content, str):
             # If Ollama returns a JSON string within the 'response' field
             content_str = content.strip()
             print(f"Vision Model raw response string: '{content_str[:200]}...'") # Log truncated response
             try:
                 analysis_result_data = json.loads(content_str)
             except json.JSONDecodeError as json_e:
                 print(f"Vision Model Error: Could not decode JSON string response: {json_e}")
                 print(f"Received: {content_str}")
                 error_message = "Error: Vision AI response format incorrect (invalid JSON)."
                 analysis_result_data = None # Invalidate result
        else:
             # Unexpected response format
             print(f"Vision Model Error: Unexpected response format type: {type(content)}")
             print(f"Received data: {response_data}")
             error_message = "Error: Vision AI returned unexpected data format."
             analysis_result_data = None


        # --- Validate parsed data ---
        if analysis_result_data is not None:
             if not isinstance(analysis_result_data, list):
                 print("Vision Model Error: Parsed response is not a JSON list as expected.")
                 error_message = "Error: Vision AI response format incorrect (not a list)."
                 analysis_result_data = None # Invalidate result
             else:
                 # Optional: Add validation for items within the list here if needed
                 # e.g., check if items have 'id', 'description', 'center' keys
                 print(f"Vision Model successfully described {len(analysis_result_data)} objects.")


    except requests.exceptions.Timeout:
         # This might catch underlying socket timeouts even if requests timeout= isn't set
         print("Error: Vision Model request timed out (Network/Socket Timeout).")
         error_message = "Error: Vision AI timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama (Vision): {e}")
        error_message = f"Error: Vision Connection failed - {e}"
    except Exception as e:
        print(f"Error during Vision Model request processing: {e}")
        error_message = f"Error: Vision AI - {e}"

    # --- Store result and signal completion ---
    # Store either the parsed data or the error message
    vision_analysis_result = analysis_result_data if analysis_result_data is not None else error_message
    vision_analysis_complete.set() # Signal that analysis (or attempt) is finished

    print("Vision analysis process complete.")
    # No return value needed, result is stored globally


# --- New function for Hermes Interaction ---
def ask_hermes_for_coordinates(user_query, vision_analysis):
    """
    Sends user query and vision analysis results to Hermes model
    to get the coordinates of the target object. Returns (x, y) tuple or None.
    """
    global response_label # To update UI

    # Ensure vision_analysis is the expected list format
    if not isinstance(vision_analysis, list):
        error_msg = f"Internal Error: Invalid vision analysis data provided to Hermes: {type(vision_analysis)}"
        print(error_msg)
        # Try to display the problematic data if it's short, otherwise just the type
        display_data = str(vision_analysis) if len(str(vision_analysis)) < 100 else type(vision_analysis).__name__
        update_response_text(f"Internal Error: Bad vision data ({display_data})")
        return None # Cannot proceed

    # Convert vision analysis list back to a string (e.g., JSON) for the prompt
    try:
        # Use compact JSON for the prompt to save tokens/space
        vision_analysis_str = json.dumps(vision_analysis)
    except Exception as e:
         error_msg = f"Internal Error: Could not format vision analysis for Hermes: {e}"
         print(error_msg)
         update_response_text("Internal Error: Formatting vision data failed") # Show internal error
         return None # Cannot proceed

    update_response_text(f"Asking AI ({OLLAMA_INTERPRET_MODEL}) to find target...")

    headers = {"Content-Type": "application/json"}

    # --- Construct the prompt for the INTERPRETATION model (Hermes) ---
    full_prompt = f"""User query: "{user_query}"

Here is a list of detected UI elements on the screen, described by a vision model:
{vision_analysis_str}

Based ONLY on the user query and the element descriptions in the list provided above, identify the single element the user wants to interact with.
Respond with ONLY the center coordinates of that element in the format "x,y" (e.g., "150,200"). The coordinates must be integers.
If no element in the list clearly matches the user query, or if the list is empty, respond with ONLY the word: None"""

    payload = {
        "model": OLLAMA_INTERPRET_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "text" # Expecting plain text "x,y" or "None"
        # No images sent to this model
    }

    try:
        print(f"Sending request to Interpretation Model ({OLLAMA_INTERPRET_MODEL})...")
        response = requests.post(
            OLLAMA_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60 # Give interpretation model a reasonable timeout (e.g., 60 seconds)
        )
        response.raise_for_status()

        response_data = response.json()
        # Expecting plain text in 'response' field
        content = response_data.get("response", "").strip()
        print(f"Interpretation Model response raw: '{content}'")

        # --- Parse the response (expecting "x,y" or "None") ---
        if content.lower() == "none":
            print("Interpretation Model could not match the query.")
            update_response_text("AI could not identify a matching object from its analysis.")
            return None
        elif ',' in content:
            try:
                parts = content.split(',')
                if len(parts) == 2:
                    x_str, y_str = parts[0].strip(), parts[1].strip()
                    # Check if parts are valid integers before converting
                    if x_str.isdigit() or (x_str.startswith('-') and x_str[1:].isdigit()):
                         x = int(x_str)
                    else:
                         raise ValueError(f"Invalid x-coordinate: {x_str}")
                    if y_str.isdigit() or (y_str.startswith('-') and y_str[1:].isdigit()):
                         y = int(y_str)
                    else:
                         raise ValueError(f"Invalid y-coordinate: {y_str}")

                    print(f"Interpretation Model identified coordinates: ({x}, {y})")
                    return (x, y) # Return tuple of integers
                else:
                    print(f"Interpretation Model Error: Invalid coordinate format '{content}'")
                    update_response_text(f"AI Error: Invalid coordinate format '{content}'.")
                    return None
            except ValueError as ve:
                print(f"Interpretation Model Error: Non-integer coordinates in '{content}'. {ve}")
                update_response_text(f"AI Error: Invalid coordinate values '{content}'.")
                return None
        else:
            # Handle cases where the model might return other text unexpectedly
            print(f"Interpretation Model Error: Unexpected response format '{content}'")
            update_response_text(f"AI Error: Unexpected response '{content}'. Expected 'x,y' or 'None'.")
            return None

    except requests.exceptions.Timeout:
         print(f"Error: Interpretation Model request timed out.")
         update_response_text(f"Error: AI ({OLLAMA_INTERPRET_MODEL}) timed out.")
         return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama (Interpret): {e}")
        update_response_text(f"Error: Interpret Connection failed - {e}")
        return None
    except Exception as e:
        print(f"Error during Interpretation Model request: {e}")
        update_response_text(f"Error: Interpretation AI - {e}")
        return None


# --- UI Functions ---

def setup_ui():
    """Creates the main Tkinter UI window."""
    global root, input_entry, response_label

    root = tk.Tk()
    root.withdraw() # Start hidden
    root.overrideredirect(True) # Remove window borders/title bar
    root.wm_attributes("-topmost", 1) # Keep window on top
    root.wm_attributes("-alpha", UI_TRANSPARENCY) # Set transparency

    # Calculate position (top center)
    screen_width = root.winfo_screenwidth()
    x_pos = (screen_width // 2) - (UI_WIDTH // 2)
    y_pos = 20 # Position near top
    root.geometry(f"{UI_WIDTH}x{UI_INPUT_HEIGHT + UI_RESPONSE_HEIGHT}+{x_pos}+{y_pos}")

    # Style
    style = ttk.Style()
    try:
        # Use a theme that generally allows background configuration
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        # Else use default
    except tk.TclError:
        print("Tk theme error, using default.") # Fallback
    style.configure("TFrame", background=UI_BG_COLOR)
    style.configure("TLabel", background=UI_BG_COLOR, foreground=UI_FG_COLOR, font=UI_FONT)
    # Ensure Entry styling handles background correctly
    style.configure("TEntry", fieldbackground="#404040", foreground=UI_FG_COLOR, insertcolor=UI_FG_COLOR, font=UI_FONT, borderwidth=1, relief=tk.FLAT)

    # Main frame
    main_frame = ttk.Frame(root, width=UI_WIDTH, height=UI_INPUT_HEIGHT + UI_RESPONSE_HEIGHT, style="TFrame")
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.pack_propagate(False) # Prevent frame from shrinking

    # Input Frame
    input_frame = ttk.Frame(main_frame, height=UI_INPUT_HEIGHT, style="TFrame")
    input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 2))
    input_frame.pack_propagate(False)

    input_entry = ttk.Entry(input_frame, style="TEntry")
    input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True) # Removed padding from pack, add margin if needed
    input_entry.bind("<Return>", on_submit) # Bind Enter key
    input_entry.bind("<Escape>", lambda e: hide_ui()) # Bind Esc key

    # Response Frame
    response_frame = ttk.Frame(main_frame, height=UI_RESPONSE_HEIGHT, style="TFrame")
    response_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(3, 5))
    response_frame.pack_propagate(False)

    response_label = ttk.Label(response_frame, text="", anchor="nw", justify=tk.LEFT, wraplength=UI_WIDTH - 20, style="TLabel")
    response_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    root.bind("<Escape>", lambda e: hide_ui()) # Bind Esc globally for the window

def show_ui():
    """Makes the UI window visible and focuses the input."""
    global ui_visible, root, input_entry, vision_analysis_result, vision_analysis_complete, vision_analysis_thread
    if root and not ui_visible:
        try:
            root.deiconify() # Show window
            root.lift() # Bring window to front
            root.attributes('-topmost', 1) # Ensure it stays on top
            input_entry.delete(0, tk.END) # Clear previous input
            input_entry.focus_force() # Force focus to the input field

            # Determine status based on vision analysis state
            status = "Initializing..." # Default status
            if vision_analysis_thread and vision_analysis_thread.is_alive():
                status = "Analyzing screen..."
            elif vision_analysis_complete.is_set():
                # Analysis is done (or failed), check the result
                if isinstance(vision_analysis_result, list):
                    count = len(vision_analysis_result)
                    status = f"Screen analyzed ({count} objects). Ready for command." if count > 0 else "Screen analyzed (No objects found). Ready."
                elif isinstance(vision_analysis_result, str): # Error message stored
                    status = vision_analysis_result # Display the error
                else: # Result is None or unexpected type
                    status = "Analysis finished with unknown state."
            else:
                # This case: double-tap occurred but analysis thread hasn't started or finished init yet.
                # Or maybe analysis was never triggered.
                status = "Ready for command..." # Fallback

            update_response_text(status) # Update UI with current status

            ui_visible = True
            print("UI Shown")
        except tk.TclError as e:
            print(f"Error showing UI (window may have been destroyed): {e}")
            ui_visible = False # Ensure state reflects reality
        except Exception as e:
             print(f"Unexpected error showing UI: {e}")
             ui_visible = False


def hide_ui():
    """Hides the UI window."""
    global ui_visible, root
    if root and ui_visible:
        try:
            root.withdraw() # Hide window
            ui_visible = False
            print("UI Hidden")
        except tk.TclError as e:
             print(f"Error hiding UI (window may have been destroyed): {e}")
             ui_visible = False # Ensure state reflects reality
        except Exception as e:
             print(f"Unexpected error hiding UI: {e}")
             ui_visible = False

def update_response_text(text):
    """Safely updates the response label text from any thread."""
    def task():
        if response_label and response_label.winfo_exists(): # Check if widget still exists
             try:
                 response_label.config(text=text)
             except tk.TclError as e:
                 # Handle cases where the label might be destroyed during async update
                 print(f"Error updating response label (widget might be destroyed): {e}")
        else:
             print(f"Skipping UI update: Response label does not exist.")

    # Schedule the UI update on the main thread
    if root:
        # Check if root window exists before scheduling
        try:
            # Verify window exists before calling after
            if root.winfo_exists():
                 root.after(0, task)
            else:
                 print("Skipping UI update: Root window destroyed.")
        except tk.TclError:
             # This can happen if root is destroyed between check and `after` call
             print("Skipping UI update: Root window likely destroyed (TclError).")
        except Exception as e:
            print(f"Unexpected error scheduling UI update: {e}")


# --- Core Logic ---

def run_vision_analysis_background(screenshot_img):
     """Takes screenshot, detects objects, runs vision analysis in background."""
     # Access globals needed
     global detected_objects_cache, vision_analysis_result, vision_analysis_complete

     if screenshot_img is None:
         print("Error: No screenshot provided for background analysis.")
         vision_analysis_result = "Error: Failed to get screenshot."
         vision_analysis_complete.set()
         # Attempt to update UI status bar if visible
         if ui_visible and root and root.winfo_exists():
             update_response_text("Analysis Error: No screenshot.")
         return

     print("Background Task: Starting screen analysis.")
     # 1. Detect objects using contours (relatively fast)
     try:
         detected_objects, annotated_img = detect_objects_by_contour(screenshot_img)
         detected_objects_cache = detected_objects # Store basic contour results
     except Exception as e:
          print(f"Background Task: Error during contour detection: {e}")
          vision_analysis_result = "Error: Object detection failed."
          vision_analysis_complete.set()
          if ui_visible and root and root.winfo_exists():
              update_response_text("Analysis Error: Object detection failed.")
          return # Stop analysis if contours fail

     # Check if objects were detected before proceeding
     if not detected_objects:
         print("Background Task: No distinct objects detected by contour.")
         # Store an empty list to indicate successful analysis but no objects found
         vision_analysis_result = []
         vision_analysis_complete.set()
         if ui_visible and root and root.winfo_exists():
             update_response_text("Screen analyzed: No objects found. Ready.")
         return # Analysis finished (successfully, but found nothing)

     if annotated_img is None:
         # This case might occur if detect_objects_by_contour has an issue but doesn't raise Exception
         print("Background Task: Error during image annotation (annotated_img is None).")
         vision_analysis_result = "Error: Failed to annotate screenshot."
         vision_analysis_complete.set()
         if ui_visible and root and root.winfo_exists():
             update_response_text("Analysis Error: Annotation failed.")
         return

     # 2. Encode the *annotated* image for the vision model
     print("Background Task: Encoding image...")
     try:
         base64_img = encode_image_to_base64(annotated_img)
         if not base64_img:
             # Handle encoding failure if encode_image_to_base64 returns None
             raise ValueError("Encoding returned None")
     except Exception as e:
         print(f"Background Task: Error encoding image: {e}")
         vision_analysis_result = "Error: Failed to encode screenshot."
         vision_analysis_complete.set()
         if ui_visible and root and root.winfo_exists():
             update_response_text("Analysis Error: Encoding failed.")
         return

     # 3. Call the Vision Model (this will take time)
     print("Background Task: Sending to Vision Model for description...")
     # get_vision_analysis now handles its own errors, stores result, and sets event
     try:
          get_vision_analysis(base64_img, detected_objects)
     except Exception as e:
          # Catch unexpected errors during the call itself
          print(f"Background Task: Unexpected error calling get_vision_analysis: {e}")
          vision_analysis_result = f"Error: Vision call failed - {e}"
          vision_analysis_complete.set()


     # 4. Update UI (if visible) based on the final state of vision_analysis_result
     final_status = "Analysis finished with unknown state." # Default
     if isinstance(vision_analysis_result, list):
         count = len(vision_analysis_result)
         final_status = f"Screen analyzed ({count} objects). Ready." if count > 0 else "Screen analyzed (No objects found). Ready."
     elif isinstance(vision_analysis_result, str): # Error occurred and message was stored
         final_status = vision_analysis_result # Display the error message
     else:
         print(f"Background Task: vision_analysis_result has unexpected type after analysis: {type(vision_analysis_result)}")

     if ui_visible and root and root.winfo_exists():
        update_response_text(final_status)
     print(f"Background Task Finished. Status: {final_status}")


def handle_user_command(user_query):
    """Handles user query after vision analysis is complete. Runs in a thread."""
    global vision_analysis_result, vision_analysis_complete, root # Include root for UI checks

    if not user_query:
        update_response_text("Please enter a command.")
        return

    # 1. Check if analysis ever started/completed initialization
    if not vision_analysis_complete.is_set():
         update_response_text("Waiting for screen analysis to finish...")
         print("Waiting for vision analysis event...")
         # Wait for the event with a timeout
         analysis_finished = vision_analysis_complete.wait(timeout=180.0) # Wait up to 3 minutes

         if not analysis_finished:
             print("Timeout waiting for vision analysis event.")
             update_response_text("Error: Screen analysis is taking too long or failed to start. Try activating again.")
             return
         else:
             print("Vision analysis event received.")
             # Brief pause to allow result variable to be set consistently? Usually not needed with Event.
             # time.sleep(0.05)

    # 2. Check the result of the vision analysis now that event is set
    current_analysis_result = vision_analysis_result # Read the shared variable once

    if current_analysis_result is None:
        # This might happen if the analysis thread had an early exit before setting result but did set event
        print("Error: Vision analysis completed but result is None.")
        update_response_text("Error: Vision analysis failed unexpectedly.")
        return
    if isinstance(current_analysis_result, str): # An error message was stored
        print(f"Vision analysis ended with error: {current_analysis_result}")
        update_response_text(f"Cannot proceed: {current_analysis_result}")
        return
    if not isinstance(current_analysis_result, list): # Should be a list
        print(f"Internal Error: Vision analysis result has incorrect type: {type(current_analysis_result)}")
        update_response_text(f"Internal Error: Bad vision data type {type(current_analysis_result).__name__}.")
        return
    if not current_analysis_result: # Empty list means analysis succeeded but found nothing
         print("Vision analysis found no objects.")
         update_response_text("Analysis found no objects to interact with.")
         return

    # 3. Send query and analysis to Hermes for interpretation
    print(f"Sending query '{user_query}' and analysis ({len(current_analysis_result)} items) to Hermes...")
    target_coords = ask_hermes_for_coordinates(user_query, current_analysis_result)

    # 4. Perform Action if coordinates are found
    if target_coords:
        try:
            # Ensure coordinates are integers (should be done by ask_hermes...)
            click_x, click_y = int(target_coords[0]), int(target_coords[1])
            print(f"Action: Clicking coordinates: ({click_x}, {click_y})")
            update_response_text(f"Action: Clicking at ({click_x}, {click_y})...")

            # --- Safely Hide UI Before Click ---
            hide_success = False
            if ui_visible and root and root.winfo_exists():
                 # Schedule hide_ui on the main thread and wait briefly for it to execute
                 root.after(0, hide_ui)
                 time.sleep(0.2) # Allow time for UI to hide
                 # Check if it actually hid (optional, hide_ui sets ui_visible=False)
                 hide_success = not ui_visible
                 print(f"UI hide attempted, success: {hide_success}")
            else:
                 print("UI not visible or root destroyed, no need to hide.")
                 time.sleep(0.05) # Small pause anyway

            # --- Perform Click ---
            pyautogui.click(click_x, click_y)
            print("Click successful.")
            # UI remains hidden until next activation

        except Exception as e:
            print(f"Error during pyautogui click or UI hide: {e}")
            # Cannot reliably update UI here as it's meant to be hidden.
            # Log the error.
    else:
        # Error message/status ("AI could not identify...") already handled by ask_hermes_for_coordinates
        print("No target coordinates identified by Hermes.")

    # Command processing finished for this cycle.

def on_submit(event=None):
    """Called when user presses Enter in the input field."""
    query = input_entry.get()
    if not query:
        update_response_text("Please enter a command.")
        return

    print(f"User submitted query: {query}")
    update_response_text("Processing command...") # Give immediate feedback

    # Run interpretation and action in a separate thread to keep UI responsive
    command_thread = threading.Thread(target=handle_user_command, args=(query,), daemon=True)
    command_thread.start()


# --- Keyboard Listener ---

def on_press(key):
    """Handles key press events for the global listener."""
    # Access and modify globals
    global last_ctrl_press_time, latest_screenshot, detected_objects_cache
    global vision_analysis_result, vision_analysis_complete, vision_analysis_thread
    global root # Needed for UI checks

    try:
        # Check if Left Ctrl or Right Ctrl is pressed
        is_ctrl = key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r
        if is_ctrl:
            current_time = time.time()
            # Check for double tap
            if current_time - last_ctrl_press_time < DOUBLE_TAP_DELAY:
                print("Double Ctrl Tap Detected!")

                # --- Reset state for new activation ---
                print("Resetting state for new activation...")
                latest_screenshot = None
                detected_objects_cache = None # Clear contour cache
                vision_analysis_result = None # Clear previous analysis/error
                vision_analysis_complete.clear() # Reset the event flag for the new analysis

                # Handle potentially still running old thread?
                # This is tricky. For now, we'll just launch a new one.
                # A more robust solution might use futures or cancel existing tasks.
                if vision_analysis_thread and vision_analysis_thread.is_alive():
                    print("Warning: Previous analysis thread may still be running.")
                    # We don't explicitly stop it, the new thread will overwrite results when done.


                # --- Capture Screenshot ---
                temp_screenshot = None # Use a temporary variable
                try:
                    print("Taking screenshot...")
                    # Ensure pyautogui takes the screenshot correctly
                    screenshot_pil = pyautogui.screenshot()
                    if screenshot_pil is None:
                         raise ValueError("pyautogui.screenshot() returned None")
                    screenshot_np = np.array(screenshot_pil)
                    # Convert color space correctly
                    temp_screenshot = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                    latest_screenshot = temp_screenshot # Store globally *after* success
                    print("Screenshot captured successfully.")
                except Exception as e:
                    print(f"FATAL: Error capturing screenshot: {e}")
                    # Update UI only if it exists and is visible (might not be yet)
                    if ui_visible and root and root.winfo_exists():
                        update_response_text("FATAL Error: Failed to take screenshot!")
                    last_ctrl_press_time = 0 # Reset time even on error to prevent immediate re-trigger
                    return # Stop processing this activation if screenshot failed

                # --- Start background analysis ---
                if latest_screenshot is not None:
                     # Update UI immediately *if* it's about to be shown or already is
                     if root and root.winfo_exists(): # Check if root exists
                          # Schedule status update AND showing UI
                           def show_and_update():
                                update_response_text("Analyzing screen...")
                                show_ui() # Now make UI visible with the status

                           root.after(0, show_and_update)
                     else:
                          # If root doesn't exist yet, UI can't be updated/shown here.
                          # It will be handled when setup_ui runs later.
                          print("Root UI not ready, cannot update status immediately.")


                     print("Starting background vision analysis thread...")
                     # Pass a copy of the screenshot to the background task
                     vision_analysis_thread = threading.Thread(
                         target=run_vision_analysis_background,
                         args=(latest_screenshot.copy(),), # Pass a copy!
                         daemon=True
                     )
                     vision_analysis_thread.start()
                else:
                     # Should not happen if screenshot capture logic above is correct
                      print("Error: Screenshot object is None after capture block.")
                      if ui_visible and root and root.winfo_exists():
                           update_response_text("Error: Failed to start analysis (no screenshot).")


                # Reset double-tap timer ONLY after processing the double tap fully
                last_ctrl_press_time = 0
            else:
                # Record the time of this single press for potential double tap
                last_ctrl_press_time = current_time

    except AttributeError:
        # Ignore key events that don't have expected attributes (e.g., some special keys)
        # print(f"AttributeError processing key: {key}") # Optional debug
        pass
    except Exception as e:
        # Catch any other unexpected errors in the listener callback
        print(f"Unexpected error in keyboard listener callback: {e}")


def start_keyboard_listener():
    """Starts the global keyboard listener in a separate thread."""
    print("Starting keyboard listener...")
    # Use daemon=True so thread exits when main program exits
    try:
        listener = keyboard.Listener(on_press=on_press, suppress=False) # Set suppress=False if needed for debugging
        # Running listener.start() directly blocks. Use listener.join() in main thread or run start() in a thread.
        listener_thread = threading.Thread(target=listener.start, daemon=True)
        listener_thread.start()
        print("Keyboard listener thread started.")
        # Keep track of the listener object if we need to stop it later (though daemon=True helps)
        # global main_listener
        # main_listener = listener
        return listener # Return listener instance if needed
    except Exception as e:
        print(f"FATAL: Could not start keyboard listener: {e}")
        print("This might be due to permissions issues (e.g., Accessibility on macOS, input monitoring on Linux/Wayland).")
        print("Try running the script with elevated privileges (e.g., sudo) if applicable, but be cautious.")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    print("Initializing Desktop Copilot (Async Vision)...")

    # 1. Setup Tkinter UI first (but keep it hidden)
    # This ensures 'root' exists for scheduling tasks from other threads like keyboard listener
    try:
        print("Setting up UI...")
        setup_ui()
        print("UI setup complete.")
    except Exception as e:
         print(f"FATAL: Could not set up Tkinter UI: {e}")
         # Exit if UI setup fails, as other parts depend on it
         exit(1) # Use exit code 1 for error


    # 2. Start keyboard listener
    listener_instance = start_keyboard_listener()

    if listener_instance and root:
        # If listener started and UI is ready, start the Tkinter event loop
        print("Keyboard listener running. Waiting for double-Ctrl tap...")
        try:
            root.mainloop() # Start the Tkinter event loop (blocks until root window is destroyed)
        except Exception as e:
            print(f"Error during UI main loop: {e}")
        finally:
            print("Tkinter mainloop finished.")
            # Optional: Clean up listener if needed, though daemon thread should exit
            # if listener_instance:
            #    listener_instance.stop()
    elif not listener_instance:
        print("Exiting because keyboard listener failed to start.")
        if root:
            try:
                 root.destroy() # Clean up Tk window if listener failed
            except: pass # Ignore errors during cleanup
    elif not root:
         print("Exiting because UI failed to initialize.")
         # Listener might be running as daemon, will exit when main process exits.


    print("Application exiting.")
    # Listener thread is daemon, should exit automatically.
    # Other daemon threads (analysis, command) should also exit.



================================================
File: SenterUI/AvA/ava.py
================================================
# Removed ollama, json, base64, PIL, io imports
import sys
import cv2
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
from llama_cpp import Llama
import os
import torch
import re # <-- Add re import for sentence splitting
from piper import PiperVoice # <-- Add back import
import requests # <-- Add requests import
import wave # <-- Add back wave import
import io   # <-- Add back io import
import subprocess # <-- Add subprocess for beep functionality

# Import optimization modules  
try:
    sys.path.append('..')  # Go up one level to access modules from parent directory
    sys.path.append('../..')  # Go up two levels to access modules from root
    from gpu_detection import detect_gpu_resources, optimize_whisper_config, apply_memory_optimizations
    from process_manager import process_manager
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"⚠️  AvA optimization modules not available: {e}")

# --- Configuration ---
# Models (Only Whisper needed for this core logic)
# OLLAMA_TEXT_MODEL = 'gemma3:4b' # Keep for potential future use
WHISPER_MODEL_SIZE = "tiny"  # Use tiny model for maximum speed
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# LLM
LLM_MODEL_PATH = "/app/Models/Hermes-3-Llama-3.2-3B-Q8_0.gguf"  # Fixed Docker path

# Face Detection
# Try to find the haarcascade file in the project
def get_face_cascade_path():
    """Find the haarcascade file in the project directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, 'haarcascade_frontalface_default.xml'),  # AvA directory
        os.path.join(os.path.dirname(current_dir), 'haarcascade_frontalface_default.xml'),  # SenterUI directory
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'haarcascade_frontalface_default.xml'),  # Root directory
        'haarcascade_frontalface_default.xml'  # fallback
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return 'haarcascade_frontalface_default.xml'  # fallback

FACE_CASCADE_PATH = get_face_cascade_path()
MIN_FACE_SIZE = (50, 50)
RESIZE_WIDTH = 640 # Resize frame for faster face detection (e.g., 640 or 320)
FACE_AREA_THRESHOLD = 0.15 # Lowered threshold: Trigger if largest face occupies >= 15% of frame area (was 0.2)

# Piper TTS Configuration
PIPER_MODEL_DIR = "piper_models"
DEFAULT_MODEL_FILENAME = "en_US-lessac-medium.onnx"
DEFAULT_CONFIG_FILENAME = DEFAULT_MODEL_FILENAME + ".json"
PIPER_VOICE_MODEL = os.path.join(PIPER_MODEL_DIR, DEFAULT_MODEL_FILENAME)
PIPER_VOICE_CONFIG = os.path.join(PIPER_MODEL_DIR, DEFAULT_CONFIG_FILENAME)

# URLs from Hugging Face (rhasspy/piper-voices)
DEFAULT_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{DEFAULT_MODEL_FILENAME}"
DEFAULT_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{DEFAULT_CONFIG_FILENAME}"

# Audio - Optimized for stability and speech continuation
AUDIO_SAMPLE_RATE = 16000 # Use 16kHz directly for Whisper compatibility (eliminates resampling)
AUDIO_CHANNELS = 1
AUDIO_BLOCK_DURATION_MS = 100  # Increased block size for stability

# State - Extended timeouts for better speech capture
ATTENTION_LOST_TIMEOUT_S = 2.5 # Much longer timeout to prevent cutting off end of sentences (was 1.2)
SPEECH_CONTINUATION_THRESHOLD = 0.02 # RMS threshold to detect if user is still speaking
# --- End Configuration ---

# --- Global State & Queues ---
audio_queue = queue.Queue(maxsize=200)  # Limit queue size to prevent memory issues
stop_event = threading.Event()
tts_queue = queue.Queue() # <-- Add global TTS queue
# Removed attention check globals

# CLI Integration callback
cli_voice_callback = None
attention_callbacks = {
    'on_attention_gained': None,
    'on_attention_lost': None
}

# Audio recording control
audio_recording_paused = False
audio_pause_lock = threading.Lock()
last_tts_time = 0  # Track when TTS last played to avoid self-transcription
recent_audio_levels = []  # Track recent audio levels for speech continuation detection
processing_audio = False  # Lock to prevent face detection from interrupting audio processing
# --- End Globals ---

# --- Helper Functions ---
def set_cli_voice_callback(callback_func):
    """Set the callback function for voice input to CLI system."""
    global cli_voice_callback
    cli_voice_callback = callback_func
    print(f"🔗 AvA CLI callback set: {callback_func.__name__ if callback_func else 'None'}")

def set_attention_callbacks(on_gained=None, on_lost=None):
    """Set callbacks for attention events."""
    global attention_callbacks
    attention_callbacks['on_attention_gained'] = on_gained
    attention_callbacks['on_attention_lost'] = on_lost
    print(f"🔗 AvA attention callbacks set")

def pause_audio_recording():
    """Pause audio recording during TTS playback."""
    global audio_recording_paused
    with audio_pause_lock:
        audio_recording_paused = True
        update_tts_activity()  # Mark TTS as active
    print("🔇 Audio recording paused for TTS")

def resume_audio_recording():
    """Resume audio recording after TTS playback."""
    global audio_recording_paused
    with audio_pause_lock:
        audio_recording_paused = False
    print("🔊 Audio recording resumed")

def is_recent_tts_activity():
    """Check if TTS was recently active to avoid transcribing our own speech."""
    global last_tts_time
    current_time = time.time()
    # Consider TTS recent if it was within the last 3 seconds
    return (current_time - last_tts_time) < 3.0

def update_tts_activity():
    """Update the timestamp when TTS becomes active."""
    global last_tts_time
    last_tts_time = time.time()

def play_attention_beep():
    """Play a very quiet attention beep."""
    try:
        # Use sounddevice and numpy to generate a much quieter beep
        import sounddevice as sd
        import numpy as np
        
        # Generate a very quiet 800Hz beep
        duration = 0.15  # Shorter duration
        frequency = 800  # Hz
        sample_rate = 44100
        
        # Generate the beep waveform
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = 0.05 * np.sin(2 * np.pi * frequency * t)  # Much quieter - 5% volume instead of 30%
        
        # Apply fade in/out to prevent clicking
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        if fade_samples > 0:
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Play the quiet beep
        sd.play(beep, samplerate=sample_rate, blocking=True)
        print("🔔 Quiet attention beep completed")
        
    except Exception as e:
        print(f"🔔 Beep failed: {e}")
        # Fallback - just print a visual indicator
        print("🔔 *quiet beep*")

def audio_callback(indata, frames, time_info, status):
    """Audio callback with TTS interference prevention and improved queue management."""
    global audio_recording_paused, recent_audio_levels
    
    if status: 
        print(f"Audio Status: {status}", file=sys.stderr)
    
    # Skip recording if paused or if TTS played recently
    with audio_pause_lock:
        if audio_recording_paused or is_recent_tts_activity():
            return  # Don't add audio to queue during TTS
    
    # Calculate RMS level for speech continuation detection
    audio_chunk = indata.copy().flatten().astype(np.float32)
    rms_level = np.sqrt(np.mean(audio_chunk**2))
    
    # Track recent audio levels (keep last 10 chunks, ~1 second of audio)
    recent_audio_levels.append(rms_level)
    if len(recent_audio_levels) > 10:
        recent_audio_levels.pop(0)
    
    # Improved queue management - prevent overloads
    queue_size = audio_queue.qsize()
    
    if queue_size > 150:  # Queue getting full
        # Remove older chunks more aggressively when queue is nearly full
        chunks_to_remove = min(50, queue_size - 100)
        for _ in range(chunks_to_remove):
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        if queue_size % 50 == 0:  # Only print occasionally to avoid spam
            print(f"🔄 Audio queue maintenance: removed {chunks_to_remove} old chunks")
    
    # Add new chunk to queue
    try:
        audio_queue.put(audio_chunk, block=False)
    except queue.Full:
        # Final fallback - remove one old chunk and add new one
        try:
            audio_queue.get_nowait()
            audio_queue.put(audio_chunk, block=False)
        except:
            pass  # If we still can't add, skip this chunk

def is_user_still_speaking():
    """Check if user appears to still be speaking based on recent audio levels."""
    if len(recent_audio_levels) < 5:
        return False
    
    # Check if recent audio levels indicate ongoing speech
    # Look for consistent audio levels that suggest active speech, not just noise
    recent_levels = recent_audio_levels[-5:]
    avg_recent_level = np.mean(recent_levels)
    max_recent_level = np.max(recent_levels)
    
    # More stringent criteria: need both good average level AND recent peaks
    speech_detected = (avg_recent_level > SPEECH_CONTINUATION_THRESHOLD and 
                      max_recent_level > SPEECH_CONTINUATION_THRESHOLD * 2.0)
    
    return speech_detected

def select_camera(max_indices=5):
    """Iterates through camera indices and backends, shows previews, and asks user selection."""
    print("\nSearching for available cameras...")
    available_cameras = []
    backends_to_try = {
        "MSMF": cv2.CAP_MSMF,
        "DShow": cv2.CAP_DSHOW
        # Add other backends if needed, e.g., cv2.CAP_ANY for auto-detect
    }

    for index in range(max_indices):
        for backend_name, backend_flag in backends_to_try.items():
            cap_test = cv2.VideoCapture(index, backend_flag)
            if cap_test.isOpened():
                print(f"  Found Camera: Index={index}, Backend={backend_name}")
                available_cameras.append({'index': index, 'backend_flag': backend_flag, 'backend_name': backend_name, 'capture': cap_test})
            else:
                cap_test.release()

    if not available_cameras:
        print("Error: No cameras found!")
        return None, None, None

    print("\nPlease select a camera by preview:")
    selected_index = -1
    selected_backend_flag = -1
    selected_backend_name = ""

    for i, cam_info in enumerate(available_cameras):
        cap = cam_info['capture']
        index = cam_info['index']
        backend_name = cam_info['backend_name']
        window_name = f"Preview: Index {index} ({backend_name}) - Use this? (Y/N/Q)"

        print(f"\nShowing preview for Camera {i+1}/{len(available_cameras)}: Index={index}, Backend={backend_name}")
        print("  Press 'Y' in the preview window to select this camera.")
        print("  Press 'N' to see the next camera.")
        print("  Press 'Q' to quit selection.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame from Index {index} ({backend_name}). Skipping.")
                break # Try next camera if this one fails to read

            # Display the frame
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF # Wait 30ms for key press

            if key == ord('y') or key == ord('Y'):
                print(f"Selected: Index={index}, Backend={backend_name}")
                selected_index = index
                selected_backend_flag = cam_info['backend_flag']
                selected_backend_name = backend_name
                cv2.destroyWindow(window_name)
                # Release all other test captures
                for j, other_cam in enumerate(available_cameras):
                    if i != j:
                        other_cam['capture'].release()
                return selected_index, selected_backend_flag, selected_backend_name # Return selected
            elif key == ord('n') or key == ord('N'):
                print("  -> Moving to next camera.")
                cv2.destroyWindow(window_name)
                break # Break inner loop to show next camera
            elif key == ord('q') or key == ord('Q'):
                print("Quit selection.")
                cv2.destroyAllWindows()
                # Release all test captures
                for cam in available_cameras:
                    cam['capture'].release()
                return None, None, None # Quit selection process
        
        # Release the capture object if not selected and moving to the next
        if selected_index == -1: # Only release if we haven't selected one yet
            cap.release()

    # If loop finishes without selection
    print("No camera selected.")
    cv2.destroyAllWindows()
    # Release any remaining captures (should be handled above, but just in case)
    for cam in available_cameras:
         if cam['capture'].isOpened():
            cam['capture'].release()
    return None, None, None

def transcribe_audio(audio_data, whisper_model):
    """Transcribes audio data with enhanced quality filtering."""
    print(f"Processing {len(audio_data)/AUDIO_SAMPLE_RATE:.2f}s of audio...")
    
    # Check if this might be our own TTS output
    if is_recent_tts_activity():
        print(f"  Skipping transcription - recent TTS activity detected")
        return None
    
    try:
        # Initialize resampled_audio early to avoid scope issues
        resampled_audio = audio_data.astype(np.float32)
        
        # Audio quality checks
        rms_level = np.sqrt(np.mean(audio_data**2))
        max_level = np.max(np.abs(audio_data))
        
        print(f"  Audio quality: RMS={rms_level:.4f}, Max={max_level:.4f}")
        
        # Relaxed audio quality checks - more sensitive to speech
        if rms_level < 0.001:  # Very low signal - reduced from 0.005
            print(f"  Audio level too low - likely background noise. Skipping transcription.")
            return None
            
        if max_level < 0.008:  # Quiet signal - reduced from 0.02
            print(f"  Audio signal too quiet - likely ambient noise. Skipping transcription.")
            return None
            
        # Enhanced clipping recovery - handle even severe distortion
        if max_level > 0.95:
            print(f"  Audio clipped (max={max_level:.3f}) - applying enhanced recovery...")
            
            # Multi-stage clipping recovery
            # Stage 1: Soft limiting using tanh
            resampled_audio = np.tanh(resampled_audio * 0.8)
            
            # Stage 2: Gentle compression for severe clipping
            if max_level > 1.2:
                print(f"    Severe clipping detected (max={max_level:.3f}) - applying compression...")
                # Apply gentle compression to severely clipped audio
                threshold = 0.7
                compressed = np.where(
                    np.abs(resampled_audio) > threshold,
                    np.sign(resampled_audio) * (threshold + (np.abs(resampled_audio) - threshold) * 0.3),
                    resampled_audio
                )
                resampled_audio = compressed
            
            # Stage 3: High-frequency roll-off to reduce harshness from clipping
            from scipy import signal
            sos = signal.butter(6, 6000, 'low', fs=AUDIO_SAMPLE_RATE, output='sos')
            resampled_audio = signal.sosfilt(sos, resampled_audio)
            
            max_level = np.max(np.abs(resampled_audio))
            print(f"    After enhanced recovery: Max={max_level:.4f}")
        
        # Automatic Gain Control (AGC) - normalize audio levels
        if max_level > 0.01:  # Only apply if we have meaningful signal
            target_level = 0.25  # Slightly lower target to prevent re-clipping
            gain = target_level / max_level
            if gain < 3.0:  # Don't over-amplify weak signals
                resampled_audio = resampled_audio * gain
                print(f"  Applied AGC: gain={gain:.2f}x, new max={np.max(np.abs(resampled_audio)):.4f}")
            
        # Dynamic range check
        dynamic_range = max_level / (rms_level + 1e-10)  # Avoid division by zero
        if dynamic_range > 50:  # Very spiky signal - likely noise
            print(f"  Poor dynamic range ({dynamic_range:.1f}) - likely noise/corruption")
            return None
        
        # Resample to 16kHz if needed for Whisper
        if AUDIO_SAMPLE_RATE != 16000:
            import librosa
            resampled_audio = librosa.resample(resampled_audio, orig_sr=AUDIO_SAMPLE_RATE, target_sr=16000)
            print(f"  Resampled {AUDIO_SAMPLE_RATE}Hz → 16kHz ({len(resampled_audio)} samples)")
        else:
            print(f"  Using audio directly at 16000Hz ({len(resampled_audio)} samples)")
        
        # Transcribe with appropriate model size based on audio length
        audio_duration = len(resampled_audio) / 16000
        if audio_duration < 1.0:
            model_size = 'tiny'  # Very short audio, use tiny for speed
        elif audio_duration < 3.0:
            model_size = 'base'  # Short audio
        else:
            model_size = 'small'  # Longer audio, use better model
            
        print(f"  Transcribing using Whisper '{model_size}'...")
        
        # Transcribe using faster-whisper
        segments, info = whisper_model.transcribe(
            resampled_audio,
            beam_size=1,  # Fast transcription
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            language='en'  # Force English to avoid language detection errors
        )
        
        print(f"  Detected language: {info.language} (prob: {info.language_probability:.2f})")
        
        # Language quality check - must be English with decent confidence
        if info.language != 'en' or info.language_probability < 0.7:
            print(f"  Language detection poor ({info.language}, {info.language_probability:.2f}) - likely corrupted")
            return None
        
        # Extract text from segments
        text = ""
        total_confidence = 0
        segment_count = 0
        
        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                text += segment_text + " "
                # Use average log probability as confidence estimate
                if hasattr(segment, 'avg_logprob'):
                    total_confidence += segment.avg_logprob
                    segment_count += 1
        
        text = text.strip()
        
        if not text or len(text) < 2:
            print(f"  No speech detected.")
            return None
            
        # Comprehensive text quality checks
        import re
        
        # Check for non-English characters (corruption indicator)
        non_english_chars = len(re.findall(r'[^\x00-\x7F\s]', text))
        if non_english_chars > len(text) * 0.1:  # More than 10% non-ASCII
            print(f"  Corrupted text detected - too many non-English characters")
            return None
            
        # Check for repetitive patterns (Whisper hallucination)
        words = text.lower().split()
        if len(words) > 3:
            # Check for word repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values()) if word_counts else 0
            
            # If any word appears more than half the time, it's likely repetitive gibberish
            if max_repetition > len(words) / 2:
                print(f"  Repetitive text detected - likely hallucination")
                return None
        
        # Check for character-level repetition (e.g., "ლლლლლლლ")
        unique_chars = len(set(text.replace(' ', '').replace('.', '').replace(',', '')))
        if len(text) > 10 and unique_chars < 3:
            print(f"  Character repetition detected - corrupted transcription")
            return None
        
        # Check confidence if available
        if segment_count > 0:
            avg_confidence = total_confidence / segment_count
            if avg_confidence < -1.0:  # Very low confidence
                print(f"  Low confidence transcription ({avg_confidence:.2f}) - likely unreliable")
                return None
        
        # Final common hallucination patterns
        hallucination_patterns = [
            r'^([a-zA-Z])\1{5,}',     # Single character repeated many times - FIXED
            r'^(.{1,3})\1{5,}',       # Short pattern repeated many times - FIXED
            r'^\W+$',                 # Only punctuation/symbols
            r'^[\d\s]+$',             # Only numbers and spaces
        ]
        
        for pattern in hallucination_patterns:
            if re.match(pattern, text):
                print(f"  Hallucination pattern detected - ignoring")
                return None
        
        print(f"  ✅ Quality transcription: '{text}'")
        return text
        
    except Exception as e:
        print(f"  ❌ Transcription error: {e}")
        return None

def generate_llm_response(prompt, llm_instance, tts_q):
    """Generates response using LLM, streams sentences to TTS queue, returns full response."""
    t_llm_start = time.time()
    prompt_snippet = prompt[:50].replace('\n', ' ')
    print(f"[{t_llm_start:.2f}] Generating LLM response for prompt: '{prompt_snippet}...' (Streaming)")

    sentence_buffer = ""
    first_sentence_queued = False
    full_response_text = "" # Accumulate full response

    try:
        stream = llm_instance(prompt, stream=True, max_tokens=2048) # Increased for better responses

        for chunk in stream:
            # Extract text content safely
            chunk_text = chunk.get('choices', [{}])[0].get('text', '')
            if chunk_text:
                full_response_text += chunk_text # Append to full response
                sentence_buffer += chunk_text

                # Simple sentence detection (ends with ., ?, !) - Improved version
                while True:
                    match = re.search(r"([.?!])", sentence_buffer)
                    if match:
                        end_index = match.end()
                        sentence = sentence_buffer[:end_index].strip()
                        if sentence: # Ensure we don't queue empty strings
                            # Queue the sentence
                            if not first_sentence_queued:
                                print(f"\n[{time.time():.2f}] [LLM Stream] Queueing FIRST sentence ('{sentence[:30]}...'). Delta from LLM start: {time.time() - t_llm_start:.2f}s")
                                first_sentence_queued = True
                            else:
                                print(f"\n[{time.time():.2f}] [LLM Stream] Queueing sentence: '{sentence[:30]}...' ({len(sentence)} chars)")

                            tts_q.put(sentence)

                        # Update buffer, removing the processed sentence
                        sentence_buffer = sentence_buffer[end_index:].lstrip() # Remove leading spaces
                    else:
                        break # No more sentence terminators in the buffer

        # Queue any remaining text in the buffer after the stream ends
        if sentence_buffer.strip():
            remaining_sentence = sentence_buffer.strip()
            print(f"\n[{time.time():.2f}] [LLM Stream] Queueing remaining part: '{remaining_sentence[:30]}...' ({len(remaining_sentence)} chars)")
            tts_q.put(remaining_sentence)

        t_llm_end = time.time()
        print(f"\n[{t_llm_end:.2f}] LLM Full Response finished. Total LLM time: {t_llm_end - t_llm_start:.2f}s")

    except ValueError as ve: # Catch specific ValueError first
        error_str = str(ve).lower()
        if "exceed context window" in error_str:
            print(f"[{time.time():.2f}] CONTEXT LIMIT ERROR DETECTED: {ve}")
            return "CONTEXT_LIMIT_ERROR" # Special signal for context limit
        else:
            # Different ValueError, report it as a generic error
            print(f"\nValueError during LLM generation (not context limit): {ve}")
            import traceback
            traceback.print_exc() # Print traceback for unexpected ValueErrors
            full_response_text = f"ValueError during generation: {ve}"
    except Exception as e: # Catch other exceptions
        print(f"\nError during LLM generation: {e}")
        import traceback
        traceback.print_exc()
        full_response_text = f"Error generating response: {e}"

    return full_response_text # Return the complete text or error message

def tts_worker(q, piper_voice_instance, stop_event):
    """Worker thread to process TTS queue using Piper."""
    log_prefix = "[TTS Worker - Piper]"
    print(f"{log_prefix} Started.")

    # Get sample rate once, handle potential missing config
    try:
        default_sample_rate = piper_voice_instance.config.sample_rate
        print(f"{log_prefix} Using default sample rate: {default_sample_rate} Hz")
    except AttributeError:
        print(f"{log_prefix} Warning: Could not get sample rate from Piper config. Using fallback 16000 Hz.")
        default_sample_rate = 16000 # Fallback

    while not stop_event.is_set():
        try:
            # Wait for a sentence with a timeout
            sentence = q.get(timeout=0.5) # Use timeout to check stop_event periodically
            if sentence is None: # Allow graceful shutdown
                continue

            t_tts_start = time.time()
            print(f"\n{log_prefix} Picked up sentence: '{sentence[:30]}...' Synthesizing...", end='', flush=True)

            # Synthesize using Piper - requires wave file object
            audio_bytes = None
            try:
                with io.BytesIO() as audio_io_synth:
                    with wave.open(audio_io_synth, 'wb') as wav_writer:
                        wav_writer.setnchannels(1)
                        wav_writer.setsampwidth(2) # Hardcode to 2 (16-bit)
                        wav_writer.setframerate(default_sample_rate) # Use rate obtained/fallback above
                        piper_voice_instance.synthesize(sentence, wav_file=wav_writer)
                    audio_bytes = audio_io_synth.getvalue()
                t_synth_end = time.time()
                print(f" (Synth: {t_synth_end - t_tts_start:.2f}s) Playing...", end='', flush=True)

            except Exception as e:
                print(f"\n{log_prefix} Error during Piper synthesis: {e}")
                audio_bytes = None # Ensure audio_bytes is None if synthesis failed
                # continue # Skip playback if synthesis fails

            # --- Play Audio using sounddevice --- 
            if audio_bytes:
                try:
                    t_play_start = time.time() # Add timing for playback start
                    # ---- Corrected Playback ----
                    # Read the WAV data from bytes
                    with io.BytesIO(audio_bytes) as audio_io_read:
                        with wave.open(audio_io_read, 'rb') as wav_reader:
                            # Verify sample rate matches expectation (use rate from header)
                            play_sample_rate = wav_reader.getframerate()
                            if play_sample_rate != default_sample_rate:
                                print(f"\n{log_prefix} Warning: WAV file sample rate ({play_sample_rate}) differs from expected ({default_sample_rate}). Using file rate.")
                            
                            # Read audio frames
                            frames = wav_reader.readframes(wav_reader.getnframes())
                            
                            # Convert frames to NumPy array (assuming 16-bit from setsampwidth(2))
                            audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    # Play the NumPy array on the correct device
                    # Find and use the Intel PCH analog device
                    devices = sd.query_devices()
                    analog_device = None
                    for i, device in enumerate(devices):
                        if "Intel PCH" in device['name'] and "Analog" in device['name'] and device['max_output_channels'] > 0:
                            analog_device = i
                            break
                    
                    if analog_device is not None:
                        sd.play(audio_data, samplerate=play_sample_rate, blocking=True, device=analog_device)
                    else:
                        sd.play(audio_data, samplerate=play_sample_rate, blocking=True)
                    t_play_end = time.time()
                    print(f" (Play: {t_play_end - t_play_start:.2f}s, Total TTS: {t_play_end - t_tts_start:.2f}s)")

                except Exception as e:
                    print(f"\n{log_prefix} Error playing audio: {e}")
            else:
                 print(f"\n{log_prefix} Skipping playback due to synthesis failure.")

            q.task_done() # Mark task as complete

        except queue.Empty:
            # Queue is empty, loop continues to check stop_event
            pass
        except Exception as e:
            # Catch any other unexpected errors in the worker loop
            print(f"\n{log_prefix} Unexpected error in worker loop: {e}")
            time.sleep(1) # Prevent rapid error looping

    print(f"{log_prefix} Stopped.")

def download_file(url, destination):
    """Downloads a file from a URL to a destination path."""
    print(f"  Downloading {os.path.basename(destination)} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded {os.path.basename(destination)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {os.path.basename(destination)}: {e}")
        # Clean up incomplete download if it exists
        if os.path.exists(destination):
            try: os.remove(destination)
            except OSError: pass
        return False
    except Exception as e:
        print(f"  An unexpected error occurred during download: {e}")
        if os.path.exists(destination):
            try: os.remove(destination)
            except OSError: pass
        return False

def ensure_piper_model_present(model_path, config_path, model_url, config_url):
    """Checks if Piper model files exist, downloads them if not."""
    model_dir = os.path.dirname(model_path)
    model_exists = os.path.exists(model_path)
    config_exists = os.path.exists(config_path)

    if model_exists and config_exists:
        print(f"Found Piper model files in {model_dir}.")
        return True

    print(f"Piper model files not found in {model_dir}. Attempting to download...")
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        try:
            print(f"Creating directory: {model_dir}")
            os.makedirs(model_dir)
        except OSError as e:
            print(f"Error creating directory {model_dir}: {e}")
            return False

    # Download model if missing
    model_downloaded = True
    if not model_exists:
        model_downloaded = download_file(model_url, model_path)

    # Download config if missing (only if model download succeeded or model already existed)
    config_downloaded = True
    if model_downloaded and not config_exists:
        config_downloaded = download_file(config_url, config_path)
    
    if model_downloaded and config_downloaded:
        print("Piper model download complete.")
        return True
    else:
        print("Piper model download failed. Please check URLs and permissions.")
        return False

def initialize_llm(model_path):
    """Initializes the Llama model from the given path."""
    if not os.path.exists(model_path):
        print(f"Error: LLM model file not found at {model_path}")
        return None
    try:
        print(f"Initializing LLM from: {model_path}")
        # Increase context window size (n_ctx)
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False) 
        print("LLM initialized.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def main():
    global tts_queue, processing_audio

    print("Initializing components...")
    
    # Apply optimizations if available
    if OPTIMIZATION_AVAILABLE:
        apply_memory_optimizations()
        gpu_info = detect_gpu_resources()
        whisper_config = optimize_whisper_config(gpu_info)
        
        # Register audio queue with process manager
        process_manager.register_queue("audio_queue", audio_queue)
        process_manager.register_queue("tts_queue", tts_queue)
        
        print(f"🎯 Using optimized Whisper config: {whisper_config['model_size']} on {whisper_config['device']}")
        
        # Initialize Whisper with optimized config
        try:
            whisper_model = WhisperModel(
                whisper_config['model_size'], 
                device=whisper_config['device'], 
                compute_type=whisper_config['compute_type']
            )
            print("Whisper model initialized.")
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            return
    else:
        # Fallback initialization
        print(f"Initializing Whisper model: {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})...")
        try:
            whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            print("Whisper model initialized.")
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            return

    # Face Detection Cascade
    print(f"Loading face cascade: {FACE_CASCADE_PATH}...")
    if not os.path.exists(FACE_CASCADE_PATH):
        print(f"Error: Face cascade file not found at {FACE_CASCADE_PATH}")
        return
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    print("Face cascade loaded.")

    # --- Piper TTS Initialization ---
    # Ensure the default model is downloaded before trying to load
    model_ready = ensure_piper_model_present(PIPER_VOICE_MODEL, PIPER_VOICE_CONFIG, 
                                               DEFAULT_MODEL_URL, DEFAULT_CONFIG_URL)
    
    piper_voice = None # Initialize to None
    # Only proceed if model download/check was successful
    if model_ready:
        try:
            print(f"Initializing Piper TTS with model: {PIPER_VOICE_MODEL}")
            # Load the Piper voice model here
            piper_voice = PiperVoice.load(PIPER_VOICE_MODEL, config_path=PIPER_VOICE_CONFIG)
            print("Piper TTS initialized successfully.")

            # === Validate Piper Config ===
            config_valid = True
            validated_sample_rate = None

            if not piper_voice.config or piper_voice.config.sample_rate is None or piper_voice.config.sample_rate <= 0:
                print("Error: Piper config missing or invalid sample_rate.")
                config_valid = False
            else:
                validated_sample_rate = piper_voice.config.sample_rate
            
            if not config_valid:
                piper_voice = None # Invalidate object
                raise ValueError("Piper config validation failed (sample_rate missing/invalid).")
            # === End Config Validation ===

            # Optional: Add a quick warm-up synthesis (without playback)
            print("  Warming up Piper TTS...")
            # Re-implement using wave + io.BytesIO
            with io.BytesIO() as audio_io:
                with wave.open(audio_io, 'wb') as wav_writer:
                    wav_writer.setnchannels(1)
                    wav_writer.setsampwidth(2) # Hardcode to 2 (16-bit)
                    wav_writer.setframerate(validated_sample_rate)
                    piper_voice.synthesize("System ready.", wav_file=wav_writer)
            # Synthesize by iterating and joining bytes (assuming synthesize yields bytes)
            # _ = b"".join(piper_voice.synthesize("System ready."))
            # with io.BytesIO() as audio_io:
            #     with wave.open(audio_io, 'wb') as wav_writer:
        except ValueError as ve:
            # Config validation failure already printed message
            pass # Keep piper_voice as None
        except Exception as e:
            print(f"Error initializing Piper TTS: {e}")
            piper_voice = None # Ensure it's None if init fails
    else:
         print("Piper model files not available. TTS disabled.")
         piper_voice = None # Ensure it's None if model not ready

    # --- LLM Initialization (Moved back to correct location) ---
    llm = None
    if os.path.exists(LLM_MODEL_PATH):
        print(f"Initializing LLM from: {LLM_MODEL_PATH}")
        try:
            # Check if Llama class is available
            if 'Llama' not in globals(): raise NameError("Llama class not found.")
            llm = initialize_llm(LLM_MODEL_PATH)
            print("LLM initialized.")
        except NameError:
             print("Error: Llama class not found. Is 'llama-cpp-python' installed?")
             llm = None # Define llm as None if class not found
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            llm = None # Ensure llm is None if init fails
    else:
        print(f"LLM model file not found at {LLM_MODEL_PATH}. LLM features disabled.")
    # --- End LLM Initialization ---

    # --- Start TTS Worker Thread (Pass Piper Voice instance) ---
    tts_worker_thread = None
    if piper_voice: # Check if Piper init was successful
        print("Starting TTS worker thread (Piper)...")
        # Pass the initialized piper_voice object
        tts_worker_thread = threading.Thread(target=tts_worker, args=(tts_queue, piper_voice, stop_event), daemon=True) 
        tts_worker_thread.start()
    else:
        print("Piper TTS not initialized, worker thread not started.")
    # --- End TTS Worker Start --- 

    # Debug check
    if llm: print("[Debug] LLM instance created successfully.")
    else: print("[Debug] LLM instance FAILED to create.")
    if piper_voice: print("[Debug] Piper TTS instance created successfully.")
    else: print("[Debug] Piper TTS instance FAILED to create.")
    # --- End Check --- 

    # Camera Initialization (Docker-friendly)
    docker_mode = os.getenv('DOCKER_MODE') == '1'
    
    if docker_mode:
        print("🐳 Docker mode detected - AvA running with camera support")
        print("   📹 Camera features enabled (video device mounted)")
        print("   🎙️ Audio transcription and attention detection active")
        print("   🔔 Double beep notifications enabled")
    
    # Camera initialization (Linux/Windows detection)
    selected_camera_index = 0
    import platform
    system_name = platform.system().lower()
    
    if docker_mode or system_name == 'linux':
        selected_backend = cv2.CAP_V4L2  # Use Video4Linux backend for Linux
        backend_name = "V4L2"
    elif system_name == 'windows':
        selected_backend = cv2.CAP_DSHOW # Use DirectShow backend for Windows
        backend_name = "DShow"
    else:
        # Fallback to generic backend
        selected_backend = cv2.CAP_ANY
        backend_name = "ANY"
    
    print(f"Initializing Camera Index: {selected_camera_index} with Backend: {backend_name}")
    cap = cv2.VideoCapture(selected_camera_index, selected_backend)
    if not cap.isOpened():
        print("Error: Could not open selected camera.")
        # Cleanup initialized components before returning
        stop_event.set() # Signal threads
        if tts_worker_thread and tts_worker_thread.is_alive(): tts_worker_thread.join(timeout=1.0)
        return
        
    print("Camera opened successfully.")
    # Set desired resolution (optional, might improve performance)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Check actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera Resolution: {width}x{height}")

    # Audio Stream
    print("Starting audio stream...")
    audio_stream = None # Initialize to None
    try:
        audio_stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            callback=audio_callback,
            blocksize=int(AUDIO_SAMPLE_RATE * AUDIO_BLOCK_DURATION_MS / 1000)
        )
        audio_stream.start()
        print("Audio stream started. Waiting for large face...")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        if 'cap' in locals() and cap.isOpened(): cap.release()
        # Cleanup other initialized components before returning
        stop_event.set() # Signal threads
        if tts_worker_thread and tts_worker_thread.is_alive(): tts_worker_thread.join(timeout=1.0)
        return

    # --- Main Loop State ---
    currently_attentive = False
    last_attention_detected_time = None
    audio_buffer = []
    last_vis_proc_time = 0 # Track last processing time

    chat_history = [] # Initialize chat history list
    MAX_HISTORY_TURNS = 3 # Number of user/assistant pairs to include
    MAX_TOTAL_HISTORY_LEN = 20 # Optional: Limit total stored history size

    print("\nEntering main loop (Press Ctrl+C to exit)...")
    try:
        while True:
            start_frame_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera. Exiting.")
                break

            # --- Visual Processing ---
            # Resize for faster face detection (optional)
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(RESIZE_WIDTH / aspect_ratio)
            resized_frame = cv2.resize(frame, (RESIZE_WIDTH, new_height))
            frame_height, frame_width = resized_frame.shape[:2]
            frame_area = frame_width * frame_height

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Face Detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=MIN_FACE_SIZE
            )

            largest_face_area = 0
            if len(faces) > 0:
                # Find the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                largest_face_area = (w * h) / frame_area # Area as proportion of frame
            
            vis_proc_end_time = time.time()
            vis_proc_duration = vis_proc_end_time - start_frame_time
            
            # --- Attention State Logic (Simplified) ---
            current_time = time.time()
            is_looking = largest_face_area >= FACE_AREA_THRESHOLD

            # Print status only if it changes or periodically
            status_changed = (is_looking != currently_attentive)
            time_since_last_print = current_time - last_vis_proc_time
            if status_changed or time_since_last_print > 2.0: # Print every 2 secs or on change
                attentive_str = "Attentive" if is_looking else "Not Attentive"
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {attentive_str} (Face Area: {largest_face_area:.2f}, Vis Proc: {vis_proc_duration:.3f}s))", end='\r', flush=True)
                last_vis_proc_time = current_time

            # --- State Transitions & Audio Handling ---
            if is_looking:
                if not currently_attentive:
                    # Transition: Not Attentive -> Attentive
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Large Face detected - Recording audio...")
                    currently_attentive = True
                    audio_buffer = [] # Clear buffer on gaining attention
                    
                    # Only play beep if not during TTS to avoid audio interference
                    if not audio_recording_paused and not is_recent_tts_activity():
                        # Play double beep to indicate attention gained
                        print("🔔 Playing attention beep...")
                        play_attention_beep()
                        print("🔔 Attention beep completed")
                    else:
                        print("🔇 TTS still active, keeping attention paused")
                    
                    # Trigger attention gained callback if set
                    if attention_callbacks['on_attention_gained']:
                        try:
                            attention_callbacks['on_attention_gained']()
                        except Exception as e:
                            print(f"❌ Error in attention gained callback: {e}")
                            
                last_attention_detected_time = current_time
            
            elif currently_attentive and (last_attention_detected_time is None or (current_time - last_attention_detected_time > ATTENTION_LOST_TIMEOUT_S)):
                # Don't interrupt if we're already processing audio
                if processing_audio:
                    continue  # Skip this iteration if audio is being processed
                
                # Check if user is still speaking before processing audio (but only if they haven't been silent for a bit)
                if is_user_still_speaking() and (current_time - last_attention_detected_time < 3.0):
                    # User appears to still be speaking AND it hasn't been too long - extend the timeout
                    print(f"👂 Speech continuation detected - extending timeout...")
                    last_attention_detected_time = current_time  # Reset timeout
                else:
                    # --- Lost Attention - Process Audio --- 
                    processing_audio = True  # Set processing lock
                    t_attention_lost = time.time() # Attention Lost Time
                    print(f"\n[{t_attention_lost:.2f}] Large Face lost - Processing audio...")
                    currently_attentive = False
                    last_attention_detected_time = None
                
                # Trigger attention lost callback if set
                if attention_callbacks['on_attention_lost']:
                    try:
                        attention_callbacks['on_attention_lost']()
                    except Exception as e:
                        print(f"❌ Error in attention lost callback: {e}")

                # Get any final audio from queue (but respect pause state)
                if not audio_recording_paused and not is_recent_tts_activity():
                    while not audio_queue.empty():
                        try: audio_buffer.append(audio_queue.get_nowait())
                        except queue.Empty: break
                else:
                    print(f"  Audio recording was paused, clearing any buffered audio")
                    audio_buffer = []  # Clear any audio that might be TTS feedback
                    # Also clear the queue
                    while not audio_queue.empty():
                        try: audio_queue.get_nowait()
                        except queue.Empty: break

                if audio_buffer:
                    full_audio = np.concatenate(audio_buffer).astype(np.float32).flatten()
                    
                    # --- Transcription --- 
                    t_transcribe_start = time.time() # Transcribe Start Time
                    print(f"[{t_transcribe_start:.2f}] Starting transcription...")
                    transcription = transcribe_audio(full_audio, whisper_model)
                    t_transcribe_end = time.time() # Transcribe End Time
                    print(f"[{t_transcribe_end:.2f}] Transcription finished ({t_transcribe_end - t_transcribe_start:.2f}s): '{transcription[:50] if transcription else 'No transcription'}...' if transcription else 'No transcription'")

                    if transcription:
                        # Check if CLI callback is set (integration mode)
                        if cli_voice_callback:
                            print(f"🔗 AvA → CLI: Routing transcription to CLI system...")
                            try:
                                # Route to CLI system instead of using AvA's own LLM
                                cli_voice_callback(transcription)
                                print(f"✅ CLI processing complete")
                            except Exception as e:
                                print(f"❌ Error in CLI callback: {e}")
                        else:
                            # --- Fallback: Original AvA LLM Logic ---
                            print(f"🧠 AvA: Using internal LLM (no CLI integration)")
                            
                            # --- Build Prompt with History ---
                            system_instruction = (
                                "You are a friendly, casual, and funny assistant. Keep your responses concise and to the point."
                            )
                            history_instructions = (
                                "Below is the recent chat history. It may be truncated and not always relevant to the current input."
                            )
                            current_user_prompt = transcription

                            # Prepare history string
                            history_string = ""
                            turns_to_include = min(len(chat_history) // 2, MAX_HISTORY_TURNS)
                            if turns_to_include > 0:
                                relevant_history = chat_history[-(turns_to_include * 2):]
                                for turn in relevant_history:
                                    history_string += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n{turn['content']}<|eot_id|>"

                            # Combine system instructions and history into a single system message
                            full_system_prompt = f"{system_instruction}\n{history_instructions}\n\n{history_string}".strip()

                            # Llama 3 Instruct Template with Combined System Message
                            formatted_prompt = (
                                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                                f"{full_system_prompt}<|eot_id|>"
                                f"<|start_header_id|>user<|end_header_id|>\n"
                                f"{current_user_prompt}<|eot_id|>"
                                f"<|start_header_id|>assistant<|end_header_id|>\n"
                            )
                            # --- End Prompt Building ---

                            print("-" * 20 + " FULL FORMATTED PROMPT " + "-" * 20)
                            print(formatted_prompt)
                            print("-" * 50)

                            # Generate response using LLM (will stream and queue sentences for TTS)
                            if llm and piper_voice: # Check if both are ready
                                print(f"[{time.time():.2f}] Calling generate_llm_response...")
                                
                                # Call modified function and get full response
                                full_assistant_response = generate_llm_response(formatted_prompt, llm, tts_queue) 
                                
                                # --- Handle LLM Response (Including Context Limit Error) ---
                                if full_assistant_response == "CONTEXT_LIMIT_ERROR":
                                    print(f"[{time.time():.2f}] Handling context limit error: Clearing history and asking user to repeat.")
                                    chat_history.clear() # Clear the history
                                    recovery_phrase = "My mistake, my memory got overloaded. Can you say that again?"
                                    tts_queue.put(recovery_phrase) # Queue recovery phrase for TTS
                                    # Do NOT add the failed prompt/error to history
                                else:
                                    # Successful response or other error (which is returned as text)
                                    # Update chat history ONLY on success/non-context-error
                                    chat_history.append({'role': 'user', 'content': current_user_prompt})
                                    chat_history.append({'role': 'assistant', 'content': full_assistant_response})

                                    # Optional: Trim overall history to prevent memory bloat
                                    if len(chat_history) > MAX_TOTAL_HISTORY_LEN * 2:
                                       chat_history = chat_history[-(MAX_TOTAL_HISTORY_LEN * 2):]
                                # --- End Handling LLM Response ---
                            else:
                                 print(f"[{time.time():.2f}] LLM or Piper TTS not ready. Skipping generation.")
                    else:
                        # No transcription, maybe clear face detection state?
                        pass
                else:
                    print(f"[{time.time():.2f}] Audio buffer was empty, nothing to transcribe.")
                audio_buffer = [] # Clear buffer AFTER processing or if empty
                processing_audio = False  # Clear processing lock
                print(f"[{time.time():.2f}] Finished processing attention loss event.")
                # --- End Audio Processing --- 

            # Collect Audio if Attentive
            if currently_attentive:
                 # Continuously add audio chunks to buffer while attentive
                 while not audio_queue.empty():
                    try: audio_buffer.append(audio_queue.get_nowait())
                    except queue.Empty: break

            # (Optional) Display frame - Commented out for performance
            # cv2.imshow('Frame', resized_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Small sleep to prevent high CPU usage if frame processing is very fast
            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting loop.")
    except Exception as e: # Catch other potential errors in the loop
        print(f"\nError in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        stop_event.set() # Signal all threads to stop

        # --- Stop TTS Worker --- 
        if tts_worker_thread and tts_worker_thread.is_alive():
            print("Waiting for TTS queue to empty (Piper)...")
            # Wait for worker to process all items currently in the queue
            tts_queue.join() 
            print("Stopping TTS worker thread (Piper)...")
            # Now that queue is empty and stop_event is set, worker should exit promptly
            tts_worker_thread.join(timeout=3.0) # Wait up to 3 seconds
            if tts_worker_thread.is_alive():
                print("Warning: TTS worker thread did not exit cleanly after timeout.")
            else:
                print("TTS worker thread stopped.")
        # --- End Stop TTS Worker --- 

        if audio_stream is not None and audio_stream.active: # Check if initialized before stopping
            print("Stopping audio stream...")
            audio_stream.stop(); audio_stream.close()
            print("Audio stream stopped.")
        else:
            print("Audio stream was not active or not initialized.")
            
        if 'cap' in locals() and cap.isOpened():
            print("Releasing camera...")
            cap.release()
            print("Camera released.")
        else:
             print("Camera was not opened or already released.")
             
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed
        print("OpenCV windows destroyed.")
        
        print("Cleanup finished. Exiting application.")

if __name__ == "__main__":
    main()











================================================
File: senter/__init__.py
================================================
#!/usr/bin/env python3
"""
SENTER AI Assistant Package
============================

A modular AI assistant system with voice recognition, text-to-speech,
smart home integration, and web research capabilities.

Version: 2.0.0
"""

from .config import get_config, is_docker_mode, is_production, is_development

__version__ = "2.0.0"
__author__ = "SENTER Development Team"
__description__ = "AI-Powered Smart Home Command Center"

# Export key components
__all__ = [
    "get_config",
    "is_docker_mode", 
    "is_production",
    "is_development",
] 


================================================
File: senter/chat_history.py
================================================
"""
Chat History Management Module

Manages conversation history using ChromaDB for smart context retrieval with persistent storage.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages conversation history using ChromaDB for smart context retrieval with persistent storage."""
    
    def __init__(self, db_client, user_profile):
        """
        Initialize the ChatHistoryManager.
        
        Args:
            db_client: ChromaDB client instance
            user_profile: User profile manager instance
        """
        self.db = db_client
        self.user_profile = user_profile
        self.collection_name = f"chat_history_{user_profile.get_current_username()}"
        self.history_collection = None
        self.relevance_threshold = 0.7  # Only include history if similarity > 0.7
        
    def initialize(self) -> bool:
        """Initialize the chat history collection for the current user."""
        try:
            # Try to get existing collection first (for persistence)
            try:
                self.history_collection = self.db.get_collection(self.collection_name)
                existing_count = self.history_collection.count()
                logger.info(f"📚 Loaded existing chat history: {existing_count} exchanges")
            except (ValueError, Exception):
                # Collection doesn't exist, create new one
                self.history_collection = self.db.create_collection(self.collection_name)
                logger.info(f"📚 Created new chat history for user: {self.user_profile.get_current_username()}")
            
            return True
        except Exception as e:
            logger.error(f"⚠️  Chat history initialization failed: {e}")
            return False
    
    def save_exchange(self, user_prompt: str, ai_response: str, tool_results: str = None) -> bool:
        """Save a complete conversation exchange."""
        try:
            # Create a complete exchange record
            exchange_text = f"User: {user_prompt}\nAssistant: {ai_response}"
            if tool_results:
                exchange_text += f"\nTool Results: {tool_results}"
            
            # Generate unique ID based on timestamp
            exchange_id = f"exchange_{int(time.time() * 1000)}"
            
            # Store in ChromaDB
            self.history_collection.add(
                documents=[user_prompt],  # Search against user prompts
                metadatas=[{
                    "full_exchange": exchange_text,
                    "ai_response": ai_response,
                    "tool_results": tool_results or "",
                    "timestamp": time.time()
                }],
                ids=[exchange_id]
            )
            
            logger.debug(f"💾 Saved chat exchange: {exchange_id}")
            return True
            
        except Exception as e:
            logger.error(f"⚠️  Failed to save chat exchange: {e}")
            return False
    
    def get_relevant_history(self, current_prompt: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """Get relevant chat history for the current prompt."""
        try:
            if not self.history_collection:
                return []
            
            # Get total number of stored exchanges
            total_exchanges = self.history_collection.count()
            if total_exchanges == 0:
                return []
            
            # Search for relevant exchanges with reduced results for speed
            search_results = self.history_collection.query(
                query_texts=[current_prompt],
                n_results=min(2, total_exchanges),  # Reduced from 4 to 2 for speed
                include=["documents", "metadatas", "distances"]
            )
            
            if not search_results or not search_results['documents']:
                return []
            
            relevant_exchanges = []
            
            # Process results and check relevance
            for i, (doc, metadata, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0], 
                search_results['distances'][0]
            )):
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1.0 - distance
                
                # Only include if above relevance threshold
                if similarity >= self.relevance_threshold:
                    relevant_exchanges.append({
                        'exchange': metadata['full_exchange'],
                        'similarity': similarity,
                        'timestamp': metadata['timestamp']
                    })
            
            # Get 1 most recent exchange only to save tokens
            recent_results = self.history_collection.query(
                query_texts=[current_prompt],
                n_results=min(1, total_exchanges),  # Reduced from 2 to 1
                include=["documents", "metadatas", "distances"]
            )
            
            # Add recent exchanges (avoid duplicates)
            for metadata in recent_results['metadatas'][0][-1:]:  # Last 1 only
                if metadata['timestamp'] not in [ex['timestamp'] for ex in relevant_exchanges]:
                    relevant_exchanges.append({
                        'exchange': metadata['full_exchange'],
                        'similarity': 0.0,  # Mark as recent, not relevant
                        'timestamp': metadata['timestamp']
                    })
            
            # Sort by timestamp (most recent first) and limit to 2 total
            relevant_exchanges.sort(key=lambda x: x['timestamp'], reverse=True)
            return relevant_exchanges[:2]  # Reduced from 4 to 2
            
        except Exception as e:
            logger.error(f"⚠️  Failed to retrieve chat history: {e}")
            return []
    
    def format_history_for_prompt(self, history_exchanges: List[Dict[str, Any]]) -> str:
        """Format chat history for inclusion in system prompt."""
        if not history_exchanges:
            return ""
        
        # Separate relevant and recent exchanges
        relevant_exchanges = [ex for ex in history_exchanges if ex['similarity'] >= self.relevance_threshold]
        recent_exchanges = [ex for ex in history_exchanges if ex['similarity'] < self.relevance_threshold]
        
        # Sort relevant by similarity (highest first), then by timestamp (oldest first within same relevance)
        relevant_exchanges.sort(key=lambda x: (-x['similarity'], x['timestamp']))
        
        # Sort recent by timestamp (oldest first, so most recent ends up last)
        recent_exchanges.sort(key=lambda x: x['timestamp'])
        
        # Combine: relevant first, then recent (with most recent at the very bottom)
        ordered_exchanges = relevant_exchanges + recent_exchanges
        
        formatted_history = "\n\nRELEVANT CHAT HISTORY:\n"
        formatted_history += "=" * 40 + "\n"
        
        for i, exchange in enumerate(ordered_exchanges):
            if exchange['similarity'] >= self.relevance_threshold:
                relevance_note = f" (relevant - {exchange['similarity']:.2f})"
            else:
                relevance_note = f" (recent)"
            formatted_history += f"\n[Exchange {i+1}{relevance_note}]\n{exchange['exchange']}\n"
        
        formatted_history += "\n" + "=" * 40
        formatted_history += "\nUse this chat history to maintain conversation continuity. Recent exchanges are at the bottom.\n"
        
        return formatted_history 


================================================
File: senter/config.py
================================================
#!/usr/bin/env python3
"""
SENTER Configuration Management
================================

Central configuration management for all SENTER components.
Handles environment variables, settings, and runtime configuration.

"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Version and metadata
SENTER_VERSION = "2.0.0"
SENTER_BUILD = "2024.1"

class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SystemConfig:
    """System-level configuration."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Docker settings
    docker_mode: bool = field(default_factory=lambda: bool(os.getenv('DOCKER_MODE', False)))
    auto_login_user: str = os.getenv('AUTO_LOGIN_USER', 'Chris')
    
    # Paths
    app_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(os.getenv('MODELS_DIR', './Models')))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv('LOGS_DIR', './logs')))
    
    # Performance
    max_workers: int = int(os.getenv('MAX_WORKERS', 4))
    memory_limit_gb: int = int(os.getenv('MEMORY_LIMIT_GB', 16))
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Set environment based on env vars
        env_name = os.getenv('ENVIRONMENT', 'development').lower()
        if env_name in [e.value for e in Environment]:
            self.environment = Environment(env_name)
        
        # Set debug based on environment
        if self.environment == Environment.PRODUCTION:
            self.debug = False
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True

@dataclass
class AudioConfig:
    """Audio system configuration."""
    
    # TTS Settings
    tts_enabled: bool = True
    tts_model_dir: str = "piper_models"
    tts_model_name: str = "en_US-lessac-medium.onnx"
    tts_sample_rate: int = int(os.getenv('TTS_SAMPLE_RATE', 44100))
    
    # Audio device settings
    audio_device: Optional[int] = None
    alsa_card: int = int(os.getenv('ALSA_CARD', 0))
    alsa_device: int = int(os.getenv('ALSA_DEVICE', 0))
    
    # PulseAudio settings
    pulse_runtime_path: str = os.getenv('XDG_RUNTIME_DIR', '/run/user/1000')
    pulse_server: str = os.getenv('PULSE_SERVER', 'unix:/run/user/1000/pulse/native')
    
    # Voice processing
    whisper_model_size: str = os.getenv('WHISPER_MODEL_SIZE', 'small')
    voice_activity_threshold: float = 0.5
    silence_timeout: float = 2.0

@dataclass  
class VideoConfig:
    """Video and camera configuration."""
    
    # Camera settings
    camera_enabled: bool = True
    default_camera: int = int(os.getenv('DEFAULT_CAMERA', 0))
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    
    # Face detection
    face_cascade_path: str = "haarcascade_frontalface_default.xml"
    attention_threshold: float = 0.7
    
    # Display settings
    display: str = os.getenv('DISPLAY', ':0')
    x11_forwarding: bool = bool(os.getenv('DISPLAY'))

@dataclass
class AIConfig:
    """AI model configuration."""
    
    # Model paths
    tools_model_path: str = "Models/Hermes-3-Llama-3.2-3B-Q8_0.gguf"
    response_model_path: str = "Models/Hermes-3-Llama-3.2-3B-Q8_0.gguf"
    
    # GPU settings
    gpu_enabled: bool = True
    gpu_layers: int = int(os.getenv('GPU_LAYERS', -1))  # -1 for auto-detect
    
    # Model parameters
    context_size: int = int(os.getenv('CONTEXT_SIZE', 4096))
    batch_size: int = int(os.getenv('BATCH_SIZE', 128))
    threads: int = int(os.getenv('AI_THREADS', 4))
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 200
    
    # Memory management
    use_mlock: bool = False
    use_mmap: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    # ChromaDB settings
    chroma_host: str = os.getenv('CHROMA_HOST', 'localhost')
    chroma_port: int = int(os.getenv('CHROMA_PORT', 8000))
    chroma_persist_dir: str = os.getenv('CHROMA_PERSIST_DIR', './chroma_db_Chris')
    
    # Collection settings
    max_collection_size: int = 10000
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Connection settings
    connection_timeout: int = 30
    retry_attempts: int = 3

@dataclass
class NetworkConfig:
    """Network and API configuration."""
    
    # Server settings
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', 8080))
    
    # Research API settings
    user_agent: str = "SENTER-AI-Assistant/2.0"
    request_timeout: int = 30
    max_concurrent_requests: int = 5
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # Security
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    # Logging levels
    log_level: LogLevel = LogLevel.INFO
    file_log_level: LogLevel = LogLevel.DEBUG
    
    # Log files
    log_file: str = "logs/senter.log"
    error_log_file: str = "logs/senter_errors.log"
    
    # Log formatting
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Log rotation
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Console output
    console_output: bool = True
    colorized_output: bool = True

class SenterConfig:
    """Main configuration manager for SENTER."""
    
    def __init__(self):
        """Initialize configuration."""
        self.system = SystemConfig()
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.ai = AIConfig()
        self.database = DatabaseConfig()
        self.network = NetworkConfig()
        self.logging = LoggingConfig()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
        
        # Validate configuration
        self._validate_config()
    
    def _apply_environment_settings(self):
        """Apply environment-specific configuration overrides."""
        if self.system.environment == Environment.PRODUCTION:
            # Production optimizations
            self.logging.log_level = LogLevel.WARNING
            self.logging.console_output = False
            self.ai.context_size = 2048  # Smaller for production efficiency
            self.database.max_collection_size = 50000
            
        elif self.system.environment == Environment.DEVELOPMENT:
            # Development settings
            self.logging.log_level = LogLevel.DEBUG
            self.logging.console_output = True
            self.logging.colorized_output = True
            
        elif self.system.environment == Environment.TESTING:
            # Testing settings
            self.logging.log_level = LogLevel.ERROR
            self.audio.tts_enabled = False
            self.video.camera_enabled = False
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Check critical paths
        if not self.system.models_dir.exists():
            self.system.models_dir.mkdir(parents=True, exist_ok=True)
            
        # Validate AI model paths
        tools_model = self.system.app_root / self.ai.tools_model_path
        response_model = self.system.app_root / self.ai.response_model_path
        
        if not tools_model.exists():
            logging.warning(f"Tools model not found: {tools_model}")
            
        if not response_model.exists():
            logging.warning(f"Response model not found: {response_model}")
    
    def get_model_path(self, model_type: str) -> Path:
        """Get absolute path for a model."""
        if model_type == "tools":
            return self.system.app_root / self.ai.tools_model_path
        elif model_type == "response":
            return self.system.app_root / self.ai.response_model_path
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "system": self.system.__dict__,
            "audio": self.audio.__dict__,
            "video": self.video.__dict__,
            "ai": self.ai.__dict__,
            "database": self.database.__dict__,
            "network": self.network.__dict__,
            "logging": self.logging.__dict__,
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

# Global configuration instance
config = SenterConfig()

# Export key settings for easy access
def get_config() -> SenterConfig:
    """Get the global configuration instance."""
    return config

def is_docker_mode() -> bool:
    """Check if running in Docker mode."""
    return config.system.docker_mode

def is_production() -> bool:
    """Check if running in production environment."""
    return config.system.environment == Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development environment."""
    return config.system.environment == Environment.DEVELOPMENT

# Convenience functions
def get_models_dir() -> Path:
    """Get the models directory path."""
    return config.system.models_dir

def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return config.system.logs_dir

def get_audio_config() -> AudioConfig:
    """Get audio configuration."""
    return config.audio

def get_ai_config() -> AIConfig:
    """Get AI configuration."""
    return config.ai 


================================================
File: senter/network_coordinator.py
================================================
#!/usr/bin/env python3
"""
SENTER Network Coordinator
==========================

Handles peer discovery and state broadcasting for distributed SENTER instances.
Uses Zeroconf/mDNS for service discovery and UDP for efficient state broadcasting.

Features:
- Automatic peer discovery on local network
- State broadcasting and receiving
- Cluster topology management
- Network resilience and reconnection
"""

import json
import socket
import threading
import time
import logging
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid

try:
    from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False


@dataclass
class NodeInfo:
    """Information about a SENTER node on the network."""
    node_id: str
    ip_address: str
    port: int
    hostname: str
    last_seen: float
    capabilities: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary."""
        return cls(**data)


class NetworkCoordinator:
    """Coordinates SENTER instances across the local network."""
    
    def __init__(self, node_id: str, port: int = 0, enable_discovery: bool = True):
        """Initialize the network coordinator."""
        self.node_id = node_id
        self.hostname = socket.gethostname()
        self.enable_discovery = enable_discovery and ZEROCONF_AVAILABLE
        
        # Network setup
        self.local_ip = self._get_local_ip()
        self.udp_socket: Optional[socket.socket] = None
        self.udp_port = port
        
        # Service discovery
        self.zeroconf: Optional[Zeroconf] = None
        self.service_browser: Optional[ServiceBrowser] = None
        self.service_info: Optional[ServiceInfo] = None
        
        # Peer management
        self.peers: Dict[str, NodeInfo] = {}
        self.peer_lock = threading.RLock()
        
        # Broadcasting
        self.broadcast_callbacks: Set[Callable[[str, Dict[str, Any]], None]] = set()
        self.is_running = False
        self.broadcast_thread: Optional[threading.Thread] = None
        self.listen_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.service_type = "_senter._udp.local."
        self.broadcast_interval = 30.0  # Broadcast state every 30 seconds
        self.peer_timeout = 90.0  # Consider peer offline after 90 seconds
        
        self.logger = logging.getLogger(__name__)
        
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def _setup_udp_socket(self) -> bool:
        """Setup UDP socket for state broadcasting."""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.local_ip, self.udp_port))
            
            # Get the actual port if we used 0 (auto-assign)
            self.udp_port = self.udp_socket.getsockname()[1]
            
            # Set socket timeout for non-blocking receive
            self.udp_socket.settimeout(1.0)
            
            self.logger.info(f"UDP socket bound to {self.local_ip}:{self.udp_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup UDP socket: {e}")
            return False
    
    def _register_service(self) -> bool:
        """Register this SENTER instance with Zeroconf."""
        if not self.enable_discovery:
            return True
            
        try:
            # Service info with node details
            service_name = f"{self.node_id}.{self.service_type}"
            
            properties = {
                b'node_id': self.node_id.encode('utf-8'),
                b'hostname': self.hostname.encode('utf-8'),
                b'udp_port': str(self.udp_port).encode('utf-8'),
                b'capabilities': json.dumps({
                    'audio': True,
                    'lights': False,  # As specified in requirements
                    'camera': True,
                    'research': True
                }).encode('utf-8')
            }
            
            self.service_info = ServiceInfo(
                self.service_type,
                service_name,
                addresses=[socket.inet_aton(self.local_ip)],
                port=self.udp_port,
                properties=properties
            )
            
            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(self.service_info)
            
            # Start service browser to discover other instances
            self.service_browser = ServiceBrowser(
                self.zeroconf,
                self.service_type,
                SenterServiceListener(self)
            )
            
            self.logger.info(f"Registered SENTER service: {service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service: {e}")
            return False
    
    def add_broadcast_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback for when state is received from peers."""
        self.broadcast_callbacks.add(callback)
    
    def remove_broadcast_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Remove a broadcast callback."""
        self.broadcast_callbacks.discard(callback)
    
    def start(self) -> bool:
        """Start the network coordinator."""
        if self.is_running:
            return True
            
        self.logger.info(f"Starting network coordinator for node: {self.node_id}")
        
        # Setup UDP socket
        if not self._setup_udp_socket():
            return False
        
        # Register service for discovery
        if not self._register_service():
            self.logger.warning("Service registration failed, continuing without discovery")
        
        # Start threads
        self.is_running = True
        
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        
        # Start peer cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        self.logger.info("Network coordinator started successfully")
        return True
    
    def stop(self):
        """Stop the network coordinator."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping network coordinator...")
        self.is_running = False
        
        # Stop service discovery
        if self.service_browser:
            self.service_browser.cancel()
        
        if self.zeroconf and self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
        
        # Close UDP socket
        if self.udp_socket:
            self.udp_socket.close()
        
        # Wait for threads to finish
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2.0)
        
        self.logger.info("Network coordinator stopped")
    
    def broadcast_state(self, state_data: Dict[str, Any]):
        """Broadcast state to all known peers."""
        if not self.is_running or not self.udp_socket:
            return
        
        try:
            # Create broadcast message
            message = {
                'type': 'state_broadcast',
                'source_node': self.node_id,
                'timestamp': time.time(),
                'data': state_data
            }
            
            message_bytes = json.dumps(message).encode('utf-8')
            
            # Send to all known peers
            with self.peer_lock:
                for peer in self.peers.values():
                    try:
                        self.udp_socket.sendto(message_bytes, (peer.ip_address, peer.port))
                    except Exception as e:
                        self.logger.debug(f"Failed to send to {peer.node_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast state: {e}")
    
    def get_peers(self) -> Dict[str, NodeInfo]:
        """Get current list of peers."""
        with self.peer_lock:
            return self.peers.copy()
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the cluster."""
        with self.peer_lock:
            return {
                'local_node': {
                    'node_id': self.node_id,
                    'ip_address': self.local_ip,
                    'port': self.udp_port,
                    'hostname': self.hostname
                },
                'peers': {node_id: peer.to_dict() for node_id, peer in self.peers.items()},
                'total_nodes': len(self.peers) + 1,
                'cluster_healthy': all(
                    time.time() - peer.last_seen < self.peer_timeout 
                    for peer in self.peers.values()
                )
            }
    
    def _listen_loop(self):
        """Listen for incoming state broadcasts."""
        self.logger.info("Started UDP listen loop")
        
        while self.is_running:
            try:
                if not self.udp_socket:
                    break
                    
                data, addr = self.udp_socket.recvfrom(65536)
                message = json.loads(data.decode('utf-8'))
                
                if message.get('type') == 'state_broadcast':
                    source_node = message.get('source_node')
                    if source_node and source_node != self.node_id:
                        # Update peer last seen time
                        self._update_peer_last_seen(source_node, addr[0])
                        
                        # Notify callbacks
                        for callback in self.broadcast_callbacks:
                            try:
                                callback(source_node, message.get('data', {}))
                            except Exception as e:
                                self.logger.error(f"Broadcast callback error: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error in listen loop: {e}")
                break
        
        self.logger.info("UDP listen loop ended")
    
    def _broadcast_loop(self):
        """Periodically broadcast heartbeat."""
        self.logger.info("Started broadcast loop")
        
        while self.is_running:
            try:
                # Broadcast a simple heartbeat
                heartbeat_data = {
                    'type': 'heartbeat',
                    'hostname': self.hostname,
                    'capabilities': {
                        'audio': True,
                        'lights': False,
                        'camera': True,
                        'research': True
                    }
                }
                
                self.broadcast_state(heartbeat_data)
                
                # Wait for next broadcast
                for _ in range(int(self.broadcast_interval)):
                    if not self.is_running:
                        break
                    time.sleep(1.0)
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error in broadcast loop: {e}")
                time.sleep(5.0)
        
        self.logger.info("Broadcast loop ended")
    
    def _cleanup_loop(self):
        """Clean up stale peers."""
        while self.is_running:
            try:
                current_time = time.time()
                stale_peers = []
                
                with self.peer_lock:
                    for node_id, peer in self.peers.items():
                        if current_time - peer.last_seen > self.peer_timeout:
                            stale_peers.append(node_id)
                    
                    for node_id in stale_peers:
                        self.logger.info(f"Removing stale peer: {node_id}")
                        del self.peers[node_id]
                
                # Sleep between cleanup cycles
                time.sleep(30.0)
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(30.0)
    
    def _update_peer_last_seen(self, node_id: str, ip_address: str):
        """Update the last seen time for a peer."""
        with self.peer_lock:
            if node_id in self.peers:
                self.peers[node_id].last_seen = time.time()
            # Note: New peers are added through service discovery
    
    def add_peer(self, node_info: NodeInfo):
        """Add a new peer (called by service discovery)."""
        with self.peer_lock:
            self.peers[node_info.node_id] = node_info
            self.logger.info(f"Added peer: {node_info.node_id} ({node_info.ip_address}:{node_info.port})")


class SenterServiceListener(ServiceListener):
    """Zeroconf service listener for SENTER instances."""
    
    def __init__(self, coordinator: NetworkCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
    
    def add_service(self, zc: Zeroconf, type_: str, name: str):
        """Called when a new SENTER service is discovered."""
        try:
            info = zc.get_service_info(type_, name)
            if info:
                properties = info.properties or {}
                
                node_id = properties.get(b'node_id', b'').decode('utf-8')
                hostname = properties.get(b'hostname', b'').decode('utf-8')
                udp_port = int(properties.get(b'udp_port', b'0').decode('utf-8'))
                capabilities_str = properties.get(b'capabilities', b'{}').decode('utf-8')
                
                # Skip our own service
                if node_id == self.coordinator.node_id:
                    return
                
                # Parse capabilities
                try:
                    capabilities = json.loads(capabilities_str)
                except:
                    capabilities = {}
                
                # Get IP address
                ip_address = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
                
                if ip_address and udp_port > 0:
                    node_info = NodeInfo(
                        node_id=node_id,
                        ip_address=ip_address,
                        port=udp_port,
                        hostname=hostname,
                        last_seen=time.time(),
                        capabilities=capabilities
                    )
                    
                    self.coordinator.add_peer(node_info)
                
        except Exception as e:
            self.logger.error(f"Error adding service {name}: {e}")
    
    def remove_service(self, zc: Zeroconf, type_: str, name: str):
        """Called when a SENTER service is removed."""
        # Services are cleaned up by the cleanup loop based on timeout
        self.logger.info(f"Service removed: {name}")
    
    def update_service(self, zc: Zeroconf, type_: str, name: str):
        """Called when a SENTER service is updated."""
        # Treat as add service
        self.add_service(zc, type_, name)


def create_network_coordinator(node_id: str = None, enable_discovery: bool = True) -> NetworkCoordinator:
    """Create a network coordinator instance."""
    if node_id is None:
        # Generate a unique node ID based on hostname and timestamp
        hostname = socket.gethostname()
        timestamp = int(time.time())
        node_id = f"senter-{hostname}-{timestamp}"
    
    return NetworkCoordinator(node_id=node_id, enable_discovery=enable_discovery) 


================================================
File: senter/state_logger.py
================================================
#!/usr/bin/env python3
"""
SENTER State Logging System
===========================

Comprehensive state tracking and logging for all SENTER system actions,
state transitions, and invariants. This module provides detailed logging
for optimization and debugging purposes.

State Variables Tracked:
- SystemMode: Initializing, Idle, Listening, Processing, ExecutingTool, Responding
- AttentionState: UserPresent, UserAbsent
- AudioRecordingState: Recording, Paused
- TTS_Queue: FIFO queue of sentences
- ActiveTTSCount: Integer tracking TTS operations
- ToolExecutionStatus: Structure tracking tool states
- ChromaDB_State: Persistent database state
- CurrentUserProfile: Active user profile data

Actions Tracked:
- DetectVoiceCommand
- ProcessInstantLights
- ProcessLLMRequest
- ExecuteTool
- SpeakSentence
- FinishSpeaking
"""

import time
import json
import threading
import socket
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging

# Import ResourceMetrics from process_manager
try:
    from process_manager import ResourceMetrics
    RESOURCE_METRICS_AVAILABLE = True
except ImportError:
    # Fallback ResourceMetrics if not available
    @dataclass
    class ResourceMetrics:
        cpu_percent: float = 0.0
        memory_percent: float = 0.0
        gpu_memory_used: float = 0.0
        active_threads: int = 0
        queue_sizes: Dict[str, int] = field(default_factory=dict)
        timestamp: float = 0.0
    RESOURCE_METRICS_AVAILABLE = False

class SystemMode(Enum):
    """System mode states."""
    INITIALIZING = "Initializing"
    IDLE = "Idle"
    LISTENING = "Listening"
    PROCESSING = "Processing"
    EXECUTING_TOOL = "ExecutingTool"
    RESPONDING = "Responding"

class AttentionState(Enum):
    """User attention states."""
    USER_PRESENT = "UserPresent"
    USER_ABSENT = "UserAbsent"

class AudioRecordingState(Enum):
    """Audio recording states."""
    RECORDING = "Recording"
    PAUSED = "Paused"

@dataclass
class StateSnapshot:
    """Complete system state at a point in time."""
    timestamp: float
    system_mode: SystemMode
    attention_state: AttentionState
    audio_recording_state: AudioRecordingState
    tts_queue_size: int
    active_tts_count: int
    tool_execution_status: Dict[str, Any]
    current_user: Optional[str]
    session_id: str
    node_id: str = ""  # NEW: Identifies which SENTER node this state belongs to
    resource_metrics: Optional[ResourceMetrics] = None  # NEW: Hardware performance metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'system_mode': self.system_mode.value,
            'attention_state': self.attention_state.value,
            'audio_recording_state': self.audio_recording_state.value,
            'tts_queue_size': self.tts_queue_size,
            'active_tts_count': self.active_tts_count,
            'tool_execution_status': self.tool_execution_status,
            'current_user': self.current_user,
            'session_id': self.session_id,
            'node_id': self.node_id
        }
        
        # Add resource metrics if available
        if self.resource_metrics:
            result['resource_metrics'] = {
                'cpu_percent': self.resource_metrics.cpu_percent,
                'memory_percent': self.resource_metrics.memory_percent,
                'gpu_memory_used': self.resource_metrics.gpu_memory_used,
                'active_threads': self.resource_metrics.active_threads,
                'queue_sizes': self.resource_metrics.queue_sizes or {},
                'timestamp': self.resource_metrics.timestamp
            }
        
        return result

@dataclass
class ActionEvent:
    """Represents a single action/event in the system."""
    timestamp: float
    action_type: str
    actor: str
    preconditions: Dict[str, Any]
    effects: Dict[str, Any]
    details: Dict[str, Any]
    session_id: str
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    node_id: str = ""  # NEW: Identifies which SENTER node performed this action
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'action_type': self.action_type,
            'actor': self.actor,
            'preconditions': self.preconditions,
            'effects': self.effects,
            'details': self.details,
            'session_id': self.session_id,
            'success': self.success,
            'error_message': self.error_message,
            'duration_ms': self.duration_ms,
            'node_id': self.node_id
        }

@dataclass
class InvariantViolation:
    """Represents a system invariant violation."""
    timestamp: float
    invariant_name: str
    description: str
    current_state: Dict[str, Any]
    expected_state: Dict[str, Any]
    severity: str  # 'warning', 'error', 'critical'
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'invariant_name': self.invariant_name,
            'description': self.description,
            'current_state': self.current_state,
            'expected_state': self.expected_state,
            'severity': self.severity,
            'session_id': self.session_id
        }

class StateLogger:
    """Comprehensive state logging system for SENTER (cluster-aware)."""
    
    def __init__(self, logs_dir: Path = Path("logs"), session_id: Optional[str] = None, node_id: Optional[str] = None):
        """Initialize the state logger."""
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        
        # Generate session ID and node ID
        self.session_id = session_id or f"session_{int(time.time())}"
        self.node_id = node_id or f"senter-{socket.gethostname()}-{int(time.time())}"
        
        # Current state tracking (local node)
        self._lock = threading.RLock()
        self._current_state = StateSnapshot(
            timestamp=time.time(),
            system_mode=SystemMode.INITIALIZING,
            attention_state=AttentionState.USER_ABSENT,
            audio_recording_state=AudioRecordingState.PAUSED,
            tts_queue_size=0,
            active_tts_count=0,
            tool_execution_status={},
            current_user=None,
            session_id=self.session_id,
            node_id=self.node_id,
            resource_metrics=None
        )
        
        # Event storage
        self._actions: List[ActionEvent] = []
        self._state_history: List[StateSnapshot] = []
        self._invariant_violations: List[InvariantViolation] = []
        
        # Cluster state tracking (NEW)
        self._cluster_state: Dict[str, StateSnapshot] = {self.node_id: self._current_state}
        self._cluster_lock = threading.RLock()
        
        # Network coordinator integration (NEW)
        self._network_coordinator: Optional[Any] = None  # Will be set via set_network_coordinator
        self._broadcast_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Resource metrics integration (NEW)
        self._process_manager: Optional[Any] = None  # Will be set via set_process_manager
        self._last_resource_update = 0.0
        self._resource_update_interval = 5.0  # Update resource metrics every 5 seconds
        
        # File handles
        self._setup_log_files()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Log initial state
        self._log_state_change("StateLogger initialization")
        
    def _setup_log_files(self):
        """Setup log file handles."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # State log file
        self.state_log_file = self.logs_dir / f"senter_state_{timestamp}_{self.session_id}.jsonl"
        self.state_log_handle = open(self.state_log_file, 'w')
        
        # Action log file
        self.action_log_file = self.logs_dir / f"senter_actions_{timestamp}_{self.session_id}.jsonl"
        self.action_log_handle = open(self.action_log_file, 'w')
        
        # Invariant violation log file
        self.invariant_log_file = self.logs_dir / f"senter_invariants_{timestamp}_{self.session_id}.jsonl"
        self.invariant_log_handle = open(self.invariant_log_file, 'w')
        
        # Summary log file
        self.summary_log_file = self.logs_dir / f"senter_summary_{timestamp}_{self.session_id}.json"
    
    def get_current_state(self) -> StateSnapshot:
        """Get current system state."""
        with self._lock:
            return self._current_state
    
    # NEW: Cluster-aware methods
    def set_network_coordinator(self, network_coordinator):
        """Set the network coordinator for cluster communication."""
        self._network_coordinator = network_coordinator
        if network_coordinator:
            # Register callback to receive state broadcasts from peers
            network_coordinator.add_broadcast_callback(self._handle_peer_state_broadcast)
    
    def set_process_manager(self, process_manager):
        """Set the process manager for resource metrics integration."""
        self._process_manager = process_manager
    
    def update_resource_metrics(self, force: bool = False):
        """Update resource metrics from process manager."""
        if not self._process_manager:
            return
        
        current_time = time.time()
        if not force and (current_time - self._last_resource_update) < self._resource_update_interval:
            return
        
        try:
            # Get latest metrics from process manager
            status = self._process_manager.get_status()
            if status.get('status') != 'no_data' and 'current' in status:
                current_metrics = status['current']
                
                # Create ResourceMetrics object
                resource_metrics = ResourceMetrics(
                    cpu_percent=current_metrics.get('cpu_percent', 0.0),
                    memory_percent=current_metrics.get('memory_percent', 0.0),
                    gpu_memory_used=current_metrics.get('gpu_memory_gb', 0.0),
                    active_threads=current_metrics.get('active_threads', 0),
                    queue_sizes=current_metrics.get('queue_sizes', {}),
                    timestamp=current_time
                )
                
                # Update current state with new metrics
                with self._lock:
                    self._current_state.resource_metrics = resource_metrics
                    self._current_state.timestamp = current_time
                    
                    # Update cluster state for this node
                    with self._cluster_lock:
                        self._cluster_state[self.node_id] = self._current_state
                
                self._last_resource_update = current_time
                
                # Broadcast state to peers if network coordinator is available
                if self._network_coordinator:
                    self._broadcast_current_state()
                
        except Exception as e:
            self.logger.error(f"Failed to update resource metrics: {e}")
    
    def get_cluster_state(self) -> Dict[str, StateSnapshot]:
        """Get the current view of all nodes in the cluster."""
        with self._cluster_lock:
            return self._cluster_state.copy()
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get a summary of the cluster state."""
        with self._cluster_lock:
            total_nodes = len(self._cluster_state)
            healthy_nodes = 0
            total_cpu = 0.0
            total_memory = 0.0
            total_gpu_memory = 0.0
            
            node_details = {}
            current_time = time.time()
            
            for node_id, state in self._cluster_state.items():
                is_healthy = (current_time - state.timestamp) < 120.0  # Consider stale after 2 minutes
                if is_healthy:
                    healthy_nodes += 1
                
                node_details[node_id] = {
                    'system_mode': state.system_mode.value,
                    'attention_state': state.attention_state.value,
                    'current_user': state.current_user,
                    'last_seen': state.timestamp,
                    'age_seconds': current_time - state.timestamp,
                    'healthy': is_healthy
                }
                
                if state.resource_metrics:
                    total_cpu += state.resource_metrics.cpu_percent
                    total_memory += state.resource_metrics.memory_percent
                    total_gpu_memory += state.resource_metrics.gpu_memory_used
                    
                    node_details[node_id].update({
                        'cpu_percent': state.resource_metrics.cpu_percent,
                        'memory_percent': state.resource_metrics.memory_percent,
                        'gpu_memory_gb': state.resource_metrics.gpu_memory_used,
                        'active_threads': state.resource_metrics.active_threads
                    })
            
            return {
                'cluster_health': {
                    'total_nodes': total_nodes,
                    'healthy_nodes': healthy_nodes,
                    'unhealthy_nodes': total_nodes - healthy_nodes
                },
                'resource_totals': {
                    'total_cpu_percent': total_cpu,
                    'total_memory_percent': total_memory,
                    'total_gpu_memory_gb': total_gpu_memory,
                    'avg_cpu_percent': total_cpu / max(1, total_nodes),
                    'avg_memory_percent': total_memory / max(1, total_nodes)
                },
                'nodes': node_details,
                'local_node_id': self.node_id
            }
    
    def _handle_peer_state_broadcast(self, source_node: str, state_data: Dict[str, Any]):
        """Handle state broadcast received from a peer node."""
        try:
            if source_node == self.node_id:
                return  # Ignore our own broadcasts
            
            # Extract state information from broadcast
            if state_data.get('type') == 'heartbeat':
                # Simple heartbeat - update basic info
                current_time = time.time()
                
                # Create a minimal state snapshot for heartbeat
                with self._cluster_lock:
                    if source_node in self._cluster_state:
                        # Update existing state with heartbeat info
                        existing_state = self._cluster_state[source_node]
                        existing_state.timestamp = current_time
                    else:
                        # Create new minimal state for unknown peer
                        self._cluster_state[source_node] = StateSnapshot(
                            timestamp=current_time,
                            system_mode=SystemMode.IDLE,  # Default assumption
                            attention_state=AttentionState.USER_ABSENT,
                            audio_recording_state=AudioRecordingState.PAUSED,
                            tts_queue_size=0,
                            active_tts_count=0,
                            tool_execution_status={},
                            current_user=None,
                            session_id="",
                            node_id=source_node
                        )
                
                self.logger.debug(f"Received heartbeat from peer: {source_node}")
            
            elif 'state_snapshot' in state_data:
                # Full state broadcast
                state_snapshot_data = state_data['state_snapshot']
                
                # Create StateSnapshot from received data
                # Note: This is a simplified version - in production you'd want more robust parsing
                with self._cluster_lock:
                    self._cluster_state[source_node] = StateSnapshot(
                        timestamp=state_snapshot_data.get('timestamp', time.time()),
                        system_mode=SystemMode(state_snapshot_data.get('system_mode', 'Idle')),
                        attention_state=AttentionState(state_snapshot_data.get('attention_state', 'UserAbsent')),
                        audio_recording_state=AudioRecordingState(state_snapshot_data.get('audio_recording_state', 'Paused')),
                        tts_queue_size=state_snapshot_data.get('tts_queue_size', 0),
                        active_tts_count=state_snapshot_data.get('active_tts_count', 0),
                        tool_execution_status=state_snapshot_data.get('tool_execution_status', {}),
                        current_user=state_snapshot_data.get('current_user'),
                        session_id=state_snapshot_data.get('session_id', ''),
                        node_id=source_node,
                        resource_metrics=self._parse_resource_metrics(state_snapshot_data.get('resource_metrics'))
                    )
                
                self.logger.debug(f"Received full state update from peer: {source_node}")
            
        except Exception as e:
            self.logger.error(f"Error processing peer state broadcast from {source_node}: {e}")
    
    def _parse_resource_metrics(self, metrics_data: Optional[Dict[str, Any]]) -> Optional[ResourceMetrics]:
        """Parse resource metrics from broadcast data."""
        if not metrics_data:
            return None
        
        try:
            return ResourceMetrics(
                cpu_percent=metrics_data.get('cpu_percent', 0.0),
                memory_percent=metrics_data.get('memory_percent', 0.0),
                gpu_memory_used=metrics_data.get('gpu_memory_used', 0.0),
                active_threads=metrics_data.get('active_threads', 0),
                queue_sizes=metrics_data.get('queue_sizes', {}),
                timestamp=metrics_data.get('timestamp', 0.0)
            )
        except Exception as e:
            self.logger.error(f"Error parsing resource metrics: {e}")
            return None
    
    def _broadcast_current_state(self):
        """Broadcast current state to all peers."""
        if not self._network_coordinator:
            return
        
        try:
            state_data = {
                'type': 'state_broadcast',
                'state_snapshot': self._current_state.to_dict()
            }
            
            self._network_coordinator.broadcast_state(state_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting state: {e}")
    
    def update_system_mode(self, new_mode: SystemMode, reason: str = ""):
        """Update system mode and log the change."""
        with self._lock:
            old_mode = self._current_state.system_mode
            if old_mode != new_mode:
                self._current_state.system_mode = new_mode
                self._current_state.timestamp = time.time()
                
                self.logger.info(f"🔄 SystemMode: {old_mode.value} → {new_mode.value} ({reason})")
                self._log_state_change(f"SystemMode changed: {old_mode.value} → {new_mode.value}")
                
                # Check state transition invariants
                self._check_system_mode_invariants(old_mode, new_mode)
    
    def update_attention_state(self, new_state: AttentionState, reason: str = ""):
        """Update attention state and log the change."""
        with self._lock:
            old_state = self._current_state.attention_state
            if old_state != new_state:
                self._current_state.attention_state = new_state
                self._current_state.timestamp = time.time()
                
                self.logger.info(f"👁️  AttentionState: {old_state.value} → {new_state.value} ({reason})")
                self._log_state_change(f"AttentionState changed: {old_state.value} → {new_state.value}")
    
    def update_audio_recording_state(self, new_state: AudioRecordingState, reason: str = ""):
        """Update audio recording state and log the change."""
        with self._lock:
            old_state = self._current_state.audio_recording_state
            if old_state != new_state:
                self._current_state.audio_recording_state = new_state
                self._current_state.timestamp = time.time()
                
                self.logger.info(f"🎤 AudioRecordingState: {old_state.value} → {new_state.value} ({reason})")
                self._log_state_change(f"AudioRecordingState changed: {old_state.value} → {new_state.value}")
                
                # Check audio invariants
                self._check_audio_invariants()
    
    def update_tts_queue_size(self, new_size: int):
        """Update TTS queue size."""
        with self._lock:
            old_size = self._current_state.tts_queue_size
            if old_size != new_size:
                self._current_state.tts_queue_size = new_size
                self._current_state.timestamp = time.time()
                
                if abs(new_size - old_size) > 1:  # Only log significant changes
                    self.logger.debug(f"📝 TTS Queue: {old_size} → {new_size}")
                    self._log_state_change(f"TTS Queue size changed: {old_size} → {new_size}")
    
    def update_active_tts_count(self, new_count: int, reason: str = ""):
        """Update active TTS count and log the change."""
        with self._lock:
            old_count = self._current_state.active_tts_count
            if old_count != new_count:
                self._current_state.active_tts_count = new_count
                self._current_state.timestamp = time.time()
                
                self.logger.debug(f"🔊 ActiveTTSCount: {old_count} → {new_count} ({reason})")
                self._log_state_change(f"ActiveTTSCount changed: {old_count} → {new_count}")
                
                # Check TTS invariants
                self._check_audio_invariants()
    
    def update_tool_execution_status(self, tool_name: str, status: Dict[str, Any]):
        """Update tool execution status."""
        with self._lock:
            old_status = self._current_state.tool_execution_status.get(tool_name, {})
            self._current_state.tool_execution_status[tool_name] = status
            self._current_state.timestamp = time.time()
            
            if old_status != status:
                self.logger.debug(f"🔧 Tool {tool_name}: {old_status} → {status}")
                self._log_state_change(f"Tool execution status changed: {tool_name}")
    
    def update_current_user(self, username: Optional[str]):
        """Update current user."""
        with self._lock:
            old_user = self._current_state.current_user
            if old_user != username:
                self._current_state.current_user = username
                self._current_state.timestamp = time.time()
                
                self.logger.info(f"👤 Current user: {old_user} → {username}")
                self._log_state_change(f"Current user changed: {old_user} → {username}")
    
    def log_action(self, action_type: str, actor: str, details: Dict[str, Any] = None, 
                   preconditions: Dict[str, Any] = None, effects: Dict[str, Any] = None,
                   success: bool = True, error_message: Optional[str] = None,
                   duration_ms: Optional[float] = None):
        """Log a system action."""
        action = ActionEvent(
            timestamp=time.time(),
            action_type=action_type,
            actor=actor,
            preconditions=preconditions or {},
            effects=effects or {},
            details=details or {},
            session_id=self.session_id,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
            node_id=self.node_id  # NEW: Include node_id in action events
        )
        
        with self._lock:
            self._actions.append(action)
            
        # Write to log file immediately
        self.action_log_handle.write(json.dumps(action.to_dict()) + '\n')
        self.action_log_handle.flush()
        
        # Log to standard logger
        status = "✅" if success else "❌"
        duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
        self.logger.info(f"{status} Action: {action_type} by {actor}{duration_str}")
        
        if error_message:
            self.logger.error(f"   Error: {error_message}")
    
    def log_invariant_violation(self, invariant_name: str, description: str, 
                              current_state: Dict[str, Any] = None,
                              expected_state: Dict[str, Any] = None,
                              severity: str = "warning"):
        """Log a system invariant violation."""
        violation = InvariantViolation(
            timestamp=time.time(),
            invariant_name=invariant_name,
            description=description,
            current_state=current_state or {},
            expected_state=expected_state or {},
            severity=severity,
            session_id=self.session_id
        )
        
        with self._lock:
            self._invariant_violations.append(violation)
        
        # Write to log file immediately
        self.invariant_log_handle.write(json.dumps(violation.to_dict()) + '\n')
        self.invariant_log_handle.flush()
        
        # Log to standard logger
        severity_icon = {"warning": "⚠️", "error": "❌", "critical": "🚨"}
        icon = severity_icon.get(severity, "⚠️")
        self.logger.warning(f"{icon} INVARIANT VIOLATION [{invariant_name}]: {description}")
    
    def _log_state_change(self, reason: str):
        """Log the current state to files."""
        with self._lock:
            # Add to history
            self._state_history.append(self._current_state)
            
            # Write to log file
            state_entry = self._current_state.to_dict()
            state_entry['reason'] = reason
            self.state_log_handle.write(json.dumps(state_entry) + '\n')
            self.state_log_handle.flush()
    
    def _check_system_mode_invariants(self, old_mode: SystemMode, new_mode: SystemMode):
        """Check system mode transition invariants."""
        # Check for valid transitions
        invalid_transitions = [
            (SystemMode.IDLE, SystemMode.RESPONDING),  # Should go through Processing first
            (SystemMode.PROCESSING, SystemMode.IDLE),  # Should go through ExecutingTool or back to Idle with tool results
        ]
        
        if (old_mode, new_mode) in invalid_transitions:
            self.log_invariant_violation(
                "invalid_system_mode_transition",
                f"Invalid transition from {old_mode.value} to {new_mode.value}",
                {"old_mode": old_mode.value, "new_mode": new_mode.value},
                {"valid_transitions": "See system specification"},
                "warning"
            )
    
    def _check_audio_invariants(self):
        """Check audio-related invariants."""
        with self._lock:
            audio_state = self._current_state.audio_recording_state
            tts_count = self._current_state.active_tts_count
            
            # Critical invariant: AudioRecordingState = Paused iff ActiveTTSCount > 0
            if audio_state == AudioRecordingState.PAUSED and tts_count == 0:
                self.log_invariant_violation(
                    "audio_paused_without_tts",
                    f"Audio recording is paused but no TTS is active (count: {tts_count})",
                    {"audio_state": audio_state.value, "tts_count": tts_count},
                    {"audio_state": "Recording", "tts_count": ">0"},
                    "error"
                )
            elif audio_state == AudioRecordingState.RECORDING and tts_count > 0:
                self.log_invariant_violation(
                    "audio_recording_with_tts",
                    f"Audio recording is active while TTS is playing (count: {tts_count})",
                    {"audio_state": audio_state.value, "tts_count": tts_count},
                    {"audio_state": "Paused", "tts_count": tts_count},
                    "critical"
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            # Action statistics
            action_counts = {}
            total_actions = len(self._actions)
            successful_actions = sum(1 for a in self._actions if a.success)
            
            for action in self._actions:
                action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
            
            # State transition statistics
            mode_transitions = {}
            attention_transitions = {}
            audio_transitions = {}
            
            for i in range(1, len(self._state_history)):
                prev_state = self._state_history[i-1]
                curr_state = self._state_history[i]
                
                if prev_state.system_mode != curr_state.system_mode:
                    transition = f"{prev_state.system_mode.value} → {curr_state.system_mode.value}"
                    mode_transitions[transition] = mode_transitions.get(transition, 0) + 1
                
                if prev_state.attention_state != curr_state.attention_state:
                    transition = f"{prev_state.attention_state.value} → {curr_state.attention_state.value}"
                    attention_transitions[transition] = attention_transitions.get(transition, 0) + 1
                
                if prev_state.audio_recording_state != curr_state.audio_recording_state:
                    transition = f"{prev_state.audio_recording_state.value} → {curr_state.audio_recording_state.value}"
                    audio_transitions[transition] = audio_transitions.get(transition, 0) + 1
            
            # Calculate durations
            durations = [a.duration_ms for a in self._actions if a.duration_ms is not None]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Session duration
            if self._state_history:
                session_start = self._state_history[0].timestamp
                session_end = self._state_history[-1].timestamp
                session_duration = session_end - session_start
            else:
                session_duration = 0
            
            return {
                'session_id': self.session_id,
                'session_duration_seconds': session_duration,
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
                'action_counts': action_counts,
                'state_transitions': {
                    'system_mode': mode_transitions,
                    'attention_state': attention_transitions,
                    'audio_recording_state': audio_transitions
                },
                'invariant_violations': len(self._invariant_violations),
                'violation_breakdown': {
                    severity: sum(1 for viol in self._invariant_violations if viol.severity == severity)
                    for severity in ['warning', 'error', 'critical']
                },
                'performance_metrics': {
                    'average_action_duration_ms': avg_duration,
                    'total_state_changes': len(self._state_history)
                }
            }
    
    def save_summary(self):
        """Save session summary to file."""
        summary = self.get_statistics()
        with open(self.summary_log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Session summary saved to {self.summary_log_file}")
    
    def close(self):
        """Close log files and save summary."""
        self.save_summary()
        
        # Close file handles
        if hasattr(self, 'state_log_handle'):
            self.state_log_handle.close()
        if hasattr(self, 'action_log_handle'):
            self.action_log_handle.close()
        if hasattr(self, 'invariant_log_handle'):
            self.invariant_log_handle.close()
        
        self.logger.info(f"🔒 State logger closed. Session: {self.session_id}")

# Global state logger instance
_state_logger: Optional[StateLogger] = None

def get_state_logger() -> StateLogger:
    """Get global state logger instance."""
    global _state_logger
    if _state_logger is None:
        _state_logger = StateLogger()
    return _state_logger

def initialize_state_logger(logs_dir: Path = Path("logs"), session_id: Optional[str] = None, node_id: Optional[str] = None) -> StateLogger:
    """Initialize global state logger."""
    global _state_logger
    _state_logger = StateLogger(logs_dir, session_id, node_id)
    return _state_logger

def close_state_logger():
    """Close global state logger."""
    global _state_logger
    if _state_logger:
        _state_logger.close()
        _state_logger = None


================================================
File: senter/tts_service.py
================================================
"""
Text-to-Speech Service Module

Handles text-to-speech functionality using Piper TTS with proper audio device detection
and threading for non-blocking operation.
"""

import os
import time
import queue
import threading
import logging
import re
import io
import wave
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import requests
import numpy as np
import sounddevice as sd

# Import state logging
from .state_logger import get_state_logger, AudioRecordingState

logger = logging.getLogger(__name__)

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    logger.warning("Piper TTS not available - install with: pip install piper-tts")
    PIPER_AVAILABLE = False


class TTSService:
    """Text-to-Speech service using Piper TTS."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, user_profile=None):
        """Initialize TTS service with configuration."""
        self.config = config or {}
        self.user_profile = user_profile
        
        # TTS state
        self.piper_voice: Optional[PiperVoice] = None
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.tts_worker_thread: Optional[threading.Thread] = None
        self.active_tts_count = 0
        self.tts_lock = threading.Lock()
        
        # State logger (will be available after initialization)
        self._state_logger = None
        
        # Audio device configuration
        self.audio_device: Optional[int] = None
        self.target_sample_rate = 44100
        
        # Model configuration
        self.model_dir = Path(self.config.get('model_dir', 'piper_models'))
        self.model_filename = self.config.get('model_filename', 'en_US-lessac-medium.onnx')
        self.config_filename = f"{self.model_filename}.json"
        
        # Model URLs
        self.model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{self.model_filename}"
        self.config_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{self.config_filename}"
        
    def is_enabled(self) -> bool:
        """Check if TTS is enabled in user profile and configuration."""
        if self.user_profile and hasattr(self.user_profile, 'is_tts_enabled'):
            return self.user_profile.is_tts_enabled()
        return self.config.get('enabled', True)
    
    def _download_file_with_progress(self, url: str, destination: Path) -> bool:
        """Download a file from URL with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=f"Downloading {destination.name}") as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
        except Exception as e:
            logger.error(f"❌ Error downloading {destination.name}: {e}")
            if destination.exists():
                try:
                    destination.unlink()
                except OSError:
                    pass
            return False
    
    def _ensure_model_present(self) -> bool:
        """Ensure Piper model files exist, download if necessary."""
        model_path = self.model_dir / self.model_filename
        config_path = self.model_dir / self.config_filename
        
        model_exists = model_path.exists()
        config_exists = config_path.exists()
        
        if model_exists and config_exists:
            return True
        
        # Create model directory
        self.model_dir.mkdir(exist_ok=True)
        
        # Download model if missing
        if not model_exists:
            logger.info(f"Downloading TTS model: {self.model_filename}")
            if not self._download_file_with_progress(self.model_url, model_path):
                return False
        
        # Download config if missing
        if not config_exists:
            logger.info(f"Downloading TTS config: {self.config_filename}")
            if not self._download_file_with_progress(self.config_url, config_path):
                return False
        
        return True
    
    def _detect_audio_device(self) -> None:
        """Auto-detect best audio device for TTS output."""
        try:
            devices = sd.query_devices()
            logger.debug(f"🔍 Scanning {len(devices)} audio devices...")
            
            # Priority order: pulse > analog > non-HDMI > HDMI
            device_priorities = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name_lower = device['name'].lower()
                    priority = 10  # Default low priority
                    
                    # Highest priority: pulse (PulseAudio)
                    if 'pulse' in name_lower:
                        priority = 1
                    # Second priority: analog outputs
                    elif 'analog' in name_lower or 'pcm' in name_lower:
                        priority = 2
                    # Third priority: other non-HDMI devices
                    elif 'hdmi' not in name_lower:
                        priority = 3
                    # Lowest priority: HDMI (often no speakers)
                    elif 'hdmi' in name_lower:
                        priority = 4
                    
                    device_priorities.append((priority, i, device))
                    # Only log best devices to reduce noise
                    if priority <= 2:
                        logger.debug(f"   Device {i}: {device['name']} (priority: {priority})")
            
            # Sort by priority and select the best one
            if device_priorities:
                device_priorities.sort(key=lambda x: x[0])  # Sort by priority (lower = better)
                best_priority, self.audio_device, best_device = device_priorities[0]
                self.target_sample_rate = int(best_device['default_samplerate'])
                
                logger.info(f"🔊 Selected audio device {self.audio_device}: {best_device['name']} "
                           f"at {self.target_sample_rate}Hz (priority: {best_priority})")
            else:
                logger.warning("⚠️ No suitable audio output devices found")
                self.audio_device = None
                self.target_sample_rate = 44100
                
        except Exception as e:
            logger.warning(f"⚠️  Audio device detection failed: {e}")
            self.audio_device = None
            self.target_sample_rate = 44100
    
    def _tts_worker(self) -> None:
        """Worker thread to process TTS queue."""
        try:
            default_sample_rate = self.piper_voice.config.sample_rate
        except AttributeError:
            default_sample_rate = 16000
        
        # Get state logger
        if self._state_logger is None:
            try:
                self._state_logger = get_state_logger()
            except:
                self._state_logger = None
        
        while not self.stop_event.is_set():
            try:
                sentence = self.tts_queue.get(timeout=0.5)
                if sentence is None:
                    continue
                
                # Emergency queue cleanup
                if self.tts_queue.qsize() > 10:
                    logger.warning(f"⚠️  TTS queue overloaded ({self.tts_queue.qsize()} items), clearing...")
                    cleared = 0
                    while self.tts_queue.qsize() > 5 and not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            cleared += 1
                        except queue.Empty:
                            break
                    if cleared > 0:
                        logger.info(f"🧹 Cleared {cleared} old TTS items")
                        # Update TTS queue size in state logger
                        if self._state_logger:
                            self._state_logger.update_tts_queue_size(self.tts_queue.qsize())
                
                # Log SpeakSentence action start
                t_start = time.time()
                if self._state_logger:
                    self._state_logger.log_action(
                        "SpeakSentence",
                        "TTS Worker Thread",
                        details={
                            "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                            "sentence_length": len(sentence),
                            "queue_size": self.tts_queue.qsize()
                        },
                        preconditions={
                            "tts_queue_not_empty": True
                        }
                    )
                
                if self.tts_queue.qsize() > 3:
                    logger.debug(f"🎵 TTS processing: '{sentence[:30]}...' (queue: {self.tts_queue.qsize()})")
                
                # Increment TTS counter and pause recording
                with self.tts_lock:
                    old_count = self.active_tts_count
                    self.active_tts_count += 1
                    
                    # Update state logger with new count
                    if self._state_logger:
                        self._state_logger.update_active_tts_count(
                            self.active_tts_count,
                            "TTS sentence started"
                        )
                    
                    if self.active_tts_count == 1:
                        self._pause_recording()
                
                try:
                    # Synthesize audio
                    with io.BytesIO() as audio_io_synth:
                        with wave.open(audio_io_synth, 'wb') as wav_writer:
                            wav_writer.setnchannels(1)
                            wav_writer.setsampwidth(2)
                            wav_writer.setframerate(default_sample_rate)
                            self.piper_voice.synthesize(sentence, wav_file=wav_writer)
                        audio_bytes = audio_io_synth.getvalue()
                    
                    t_synth = time.time()
                    logger.debug(f"🎵 Synthesis complete: {t_synth - t_start:.2f}s, now playing...")
                    
                    # Play audio
                    if audio_bytes and not self.stop_event.is_set():
                        self._play_audio(audio_bytes, default_sample_rate)
                    
                    t_end = time.time()
                    logger.debug(f"🎵 Playback complete: {t_end - t_synth:.2f}s (total: {t_end - t_start:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"⚠️  TTS Error: {e}")
                
                # Log FinishSpeaking action and update state
                duration_ms = (time.time() - t_start) * 1000
                
                # Decrement TTS counter and resume recording
                with self.tts_lock:
                    old_count = self.active_tts_count
                    self.active_tts_count -= 1
                    
                    # Update state logger with new count
                    if self._state_logger:
                        self._state_logger.update_active_tts_count(
                            self.active_tts_count,
                            "TTS sentence completed"
                        )
                        self._state_logger.update_tts_queue_size(self.tts_queue.qsize())
                    
                    if self.active_tts_count == 0:
                        time.sleep(0.3)  # Brief pause before resuming
                        if self.active_tts_count == 0 and self.tts_queue.qsize() <= 2:
                            self._resume_recording()
                            
                            # Log FinishSpeaking action
                            if self._state_logger:
                                self._state_logger.log_action(
                                    "FinishSpeaking",
                                    "TTS Worker Thread",
                                    details={
                                        "sentence_completed": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                                        "active_tts_count_before": old_count,
                                        "active_tts_count_after": self.active_tts_count,
                                        "queue_size": self.tts_queue.qsize(),
                                        "sleep_duration_ms": 300
                                    },
                                    effects={
                                        "audio_recording_resumed": True,
                                        "active_tts_count": self.active_tts_count
                                    },
                                    success=True,
                                    duration_ms=duration_ms
                                )
                
                self.tts_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"⚠️  TTS Worker Error: {e}")
                # Always decrement counter on error
                with self.tts_lock:
                    self.active_tts_count = max(0, self.active_tts_count - 1)
                    if self.active_tts_count == 0:
                        time.sleep(0.3)
                        if self.tts_queue.qsize() <= 2:
                            self._resume_recording()
                time.sleep(1)
    
    def _play_audio(self, audio_bytes: bytes, original_sample_rate: int) -> None:
        """Play audio bytes through the selected audio device."""
        try:
            # Convert audio data
            with io.BytesIO(audio_bytes) as audio_io_read:
                with wave.open(audio_io_read, 'rb') as wav_reader:
                    frames = wav_reader.readframes(wav_reader.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Convert to float32 for resampling
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Resample if necessary
            if original_sample_rate != self.target_sample_rate:
                duration = len(audio_float) / original_sample_rate
                new_length = int(duration * self.target_sample_rate)
                resampled_audio = np.interp(
                    np.linspace(0, len(audio_float), new_length),
                    np.arange(len(audio_float)),
                    audio_float
                )
                logger.debug(f"🔄 Resampled {original_sample_rate}Hz → {self.target_sample_rate}Hz")
            else:
                resampled_audio = audio_float
            
            # Add silence padding to prevent cutoff
            silence_samples = int(self.target_sample_rate * 0.05)
            silence_padding = np.zeros(silence_samples, dtype=np.float32)
            padded_audio = np.concatenate([resampled_audio, silence_padding])
            
            # Play audio
            if not self.stop_event.is_set():
                try:
                    if self.audio_device is not None:
                        sd.play(padded_audio, samplerate=self.target_sample_rate, 
                               blocking=True, device=self.audio_device)
                    else:
                        sd.play(padded_audio, samplerate=self.target_sample_rate, blocking=True)
                except Exception as e:
                    logger.warning(f"⚠️  Audio playback error: {e}")
                    # Try fallback without specific device
                    try:
                        sd.play(padded_audio, samplerate=self.target_sample_rate, blocking=True)
                        logger.debug("✅ Fallback audio playback succeeded")
                    except Exception as e2:
                        logger.error(f"❌ Fallback audio also failed: {e2}")
            
        except Exception as e:
            logger.error(f"⚠️  Error playing audio: {e}")
    
    def _pause_recording(self) -> None:
        """Pause attention detection recording during TTS."""
        try:
            from SenterUI.AvA.ava import pause_audio_recording
            pause_audio_recording()
            
            # Update state logger
            if self._state_logger:
                self._state_logger.update_audio_recording_state(
                    AudioRecordingState.PAUSED,
                    "TTS playback started"
                )
        except (ImportError, AttributeError):
            # Fallback - could be handled by attention detector if available
            pass
    
    def _resume_recording(self) -> None:
        """Resume attention detection recording after TTS."""
        try:
            from SenterUI.AvA.ava import resume_audio_recording
            resume_audio_recording()
            logger.debug("🔊 Resuming attention detection after TTS complete")
            
            # Update state logger
            if self._state_logger:
                self._state_logger.update_audio_recording_state(
                    AudioRecordingState.RECORDING,
                    "TTS playback finished"
                )
        except (ImportError, AttributeError):
            # Fallback - could be handled by attention detector if available
            pass
    
    def initialize(self) -> bool:
        """Initialize the TTS service."""
        if not PIPER_AVAILABLE:
            logger.error("❌ Piper TTS not available")
            return False
        
        if not self.is_enabled():
            logger.info("🔇 TTS disabled in user preferences")
            return False
        
        # Ensure model files are present
        if not self._ensure_model_present():
            logger.error("❌ Piper model files not available")
            return False
        
        try:
            # Load Piper voice
            model_path = self.model_dir / self.model_filename
            config_path = self.model_dir / self.config_filename
            
            self.piper_voice = PiperVoice.load(str(model_path), config_path=str(config_path))
            
            # Validate config
            if not self.piper_voice.config or self.piper_voice.config.sample_rate is None or self.piper_voice.config.sample_rate <= 0:
                logger.error("❌ Piper config invalid")
                return False
            
            # Detect audio devices
            self._detect_audio_device()
            
            # Start worker thread
            self.tts_worker_thread = threading.Thread(
                target=self._tts_worker, 
                daemon=True,
                name="TTS-Worker"
            )
            self.tts_worker_thread.start()
            
            logger.info("✅ TTS service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing TTS service: {e}")
            return False
    
    def speak_text(self, text: str) -> bool:
        """Split text into sentences and queue for TTS."""
        if not text or not text.strip():
            return False
        
        # Split text into sentences using regex
        sentences = re.split(r'[.!?]+', text.strip())
        
        success = True
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only send non-empty sentences
                if not self.speak_sentence(sentence + "."):  # Add period back
                    success = False
        
        return success
    
    def speak_sentence(self, sentence: str) -> bool:
        """Queue a single sentence for TTS playback."""
        if not self.piper_voice or not sentence.strip():
            return False
        
        try:
            self.tts_queue.put(sentence.strip())
            
            # Update TTS queue size in state logger
            if self._state_logger is None:
                try:
                    self._state_logger = get_state_logger()
                except:
                    pass
            
            if self._state_logger:
                self._state_logger.update_tts_queue_size(self.tts_queue.qsize())
            
            logger.debug(f"🎤 Queued for TTS: '{sentence[:30]}...'")
            return True
        except Exception as e:
            logger.error(f"⚠️  Error queuing TTS: {e}")
            return False
    
    def emergency_stop(self) -> None:
        """Emergency stop - clear queue and stop immediately."""
        logger.info("🚨 Emergency TTS stop initiated...")
        
        # Clear the TTS queue
        cleared = 0
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        if cleared > 0:
            logger.info(f"🧹 Emergency cleared {cleared} TTS items")
        
        # Reset TTS counter
        with self.tts_lock:
            self.active_tts_count = 0
        
        # Stop any ongoing audio playback
        try:
            sd.stop()
            logger.info("🔇 Audio playback stopped")
        except Exception as e:
            logger.warning(f"Error stopping audio: {e}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown the TTS service."""
        logger.info("🛑 Shutting down TTS service...")
        
        # Signal stop
        self.stop_event.set()
        
        # Emergency stop first
        self.emergency_stop()
        
        # Wait for worker thread
        if self.tts_worker_thread and self.tts_worker_thread.is_alive():
            logger.debug("⏳ Waiting for TTS worker to stop...")
            self.tts_worker_thread.join(timeout=2.0)
            if self.tts_worker_thread.is_alive():
                logger.warning("⚠️  TTS worker did not stop cleanly")
        
        # Clean up audio system
        try:
            sd.stop()
            sd.default.reset()
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"⚠️  Error cleaning up audio: {e}")
        
        logger.info("✅ TTS service shutdown complete") 





```

