# Repository Analysis

## Summary

```
Directory: home/ath/SENTER
Files analyzed: 12

Estimated tokens: 36.1k
```

## Important Files

```
Directory structure:
└── SENTER/
    ├── ava_audio_config.py
    ├── gpu_detection.py
    ├── journal_system.py
    ├── main_v2.py
    ├── process_manager.py
    ├── tools_config.py
    ├── user_profiles.py
    ├── Documentation/
    ├── Models/
    ├── QUARANTINE_AVA_DO_NOT_TOUCH/
    ├── SenterUI/
    ├── chroma_db_Chris/
    ├── chroma_db_Chris_Chris/
    ├── chroma_db_temp/
    ├── logs/
    ├── piper_models/
    ├── senter/
    │   ├── __init__.py
    │   ├── chat_history.py
    │   ├── config.py
    │   ├── state_logger.py
    │   ├── tts_service.py
    │   └── __pycache__/
    ├── user_profiles/
    └── whisper_models/

```

## Content

```
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
File: main_v2.py
================================================
#!/usr/bin/env python3
"""
SENTER - AI-Powered Smart Home Command Center (Version 2.0)
============================================================

🐳 DOCKER CONTAINER EXECUTION ONLY 🐳
This script is designed to run exclusively inside a Docker container.

DO NOT run this script directly on the host system.
Use: docker-compose exec senter python main_v2.py

For container management:
- Start: docker-compose up -d
- Stop: docker-compose down  
- Logs: docker-compose logs -f senter
- Shell: docker-compose exec senter /bin/bash

All dependencies, device access, and environment setup
are handled by the Docker container environment.
"""

import os
import sys
import time
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# Fix OpenMP conflict FIRST
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import new configuration system
from senter.config import get_config, is_docker_mode, is_production

# Import existing modules (keeping compatibility)
from user_profiles import UserProfile
from SenterUI.ui_components import SenterUI
from light_controller import execute_light_command, set_user_credentials
from tools_config import get_formatted_tools_list
from research import execute_research
from journal_system import initialize_journal_system, add_interaction_to_journal

# Import state logging system
from senter.state_logger import (
    StateLogger, SystemMode, AttentionState, AudioRecordingState,
    initialize_state_logger, get_state_logger, close_state_logger
)

# Import optimization modules
try:
    from gpu_detection import detect_gpu_resources, optimize_llm_config
    from process_manager import init_process_management
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️  Optimization modules not available.")

# Import camera tools (optional)
try:
    from camera_tools import execute_camera_command
    CAMERA_TOOLS_AVAILABLE = True
except ImportError:
    CAMERA_TOOLS_AVAILABLE = False
    print("⚠️  Camera tools not available.")

class SenterApplication:
    """Main SENTER application class."""
    
    def __init__(self):
        """Initialize the SENTER application."""
        self.config = get_config()
        self.user_profile: Optional[UserProfile] = None
        self.senter_tools = None
        self.senter_response = None
        self.db = None
        self.chat_history_manager = None
        self.attention_detector = None
        self.tts_system = None
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state logger
        self.state_logger = initialize_state_logger(
            logs_dir=self.config.system.logs_dir,
            session_id=f"senter_{int(time.time())}"
        )
        
        # Verify Docker environment
        self._verify_docker_environment()
    
    def _setup_logging(self):
        """Setup centralized logging."""
        log_config = self.config.logging
        
        # Create logs directory
        self.config.system.logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.log_level.value),
            format=log_config.log_format,
            datefmt=log_config.date_format,
            handlers=[
                logging.FileHandler(log_config.log_file),
                logging.StreamHandler() if log_config.console_output else logging.NullHandler()
            ]
        )
        
        # Suppress verbose logs from external libraries
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        logging.getLogger('faster_whisper').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SENTER v{self.config.system.app_root} starting...")
    
    def _verify_docker_environment(self):
        """Verify we're running in Docker container."""
        if not is_docker_mode() and not os.path.exists('/.dockerenv'):
            self.logger.error("🚨 ERROR: SENTER must run inside Docker container!")
            print("\n🚨 ERROR: SENTER must run inside Docker container!")
            print("\nThis script is designed for Docker container execution only.")
            print("Please use one of these commands:")
            print("\n📋 Start container:     docker-compose up -d")
            print("🚀 Run SENTER:         docker-compose exec senter python main_v2.py")
            print("📊 Container logs:     docker-compose logs -f senter")
            print("🐚 Container shell:    docker-compose exec senter /bin/bash")
            print("\nFor more help, see: phone_setup_guide.md")
            sys.exit(1)
    
    def _initialize_user_profile(self) -> bool:
        """Initialize user profile system."""
        try:
            self.state_logger.log_action(
                "InitializeUserProfile", 
                "SenterApplication",
                details={"step": "start"}
            )
            
            self.user_profile = UserProfile()
            self.user_profile.setup_initial_profiles()
            
            # Auto-login for Docker environment
            if is_docker_mode():
                username = self.config.system.auto_login_user
                if not self.user_profile.login(username, ""):
                    self.logger.error(f"Auto-login failed for user: {username}")
                    self.state_logger.log_action(
                        "InitializeUserProfile",
                        "SenterApplication", 
                        success=False,
                        error_message=f"Auto-login failed for user: {username}"
                    )
                    return False
                self.logger.info(f"Auto-logged in as: {username}")
                self.state_logger.update_current_user(username)
            
            # Set user credentials for lights
            set_user_credentials(self.user_profile.get_current_user_data())
            
            self.state_logger.log_action(
                "InitializeUserProfile",
                "SenterApplication",
                details={"step": "complete", "user": self.user_profile.get_current_username()},
                success=True
            )
            return True
            
        except Exception as e:
            self.logger.error(f"User profile initialization failed: {e}")
            self.state_logger.log_action(
                "InitializeUserProfile",
                "SenterApplication",
                success=False,
                error_message=str(e)
            )
            return False
    
    def _initialize_ai_models(self) -> bool:
        """Initialize AI models with optimized configuration."""
        try:
            if OPTIMIZATION_AVAILABLE:
                gpu_info = detect_gpu_resources()
                llm_config = optimize_llm_config(gpu_info)
            else:
                # Fallback configuration
                llm_config = {
                    'n_gpu_layers': self.config.ai.gpu_layers,
                    'n_ctx': self.config.ai.context_size,
                    'n_batch': self.config.ai.batch_size,
                    'n_threads': self.config.ai.threads,
                    'use_mlock': self.config.ai.use_mlock,
                    'use_mmap': self.config.ai.use_mmap,
                    'verbose': False
                }
            
            # Import llama-cpp-python here to avoid import issues
            from llama_cpp import Llama
            
            # Get model paths
            tools_model = self.config.get_model_path("tools")
            response_model = self.config.get_model_path("response")
            
            if not tools_model.exists():
                self.logger.error(f"Tools model not found: {tools_model}")
                return False
                
            if not response_model.exists():
                self.logger.error(f"Response model not found: {response_model}")
                return False
            
            self.logger.info("🧠 Loading AI models...")
            
            # Load models
            self.senter_tools = Llama(model_path=str(tools_model), **llm_config)
            self.senter_response = Llama(model_path=str(response_model), **llm_config)
            
            self.logger.info("✅ AI models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"AI model initialization failed: {e}")
            return False
    
    def _initialize_database(self) -> bool:
        """Initialize ChromaDB with user-specific configuration."""
        try:
            from chromadb import PersistentClient
            
            # Use configuration settings
            db_config = self.config.database
            username = self.user_profile.get_current_username()
            persist_dir = f"{db_config.chroma_persist_dir}_{username}"
            
            self.db = PersistentClient(path=persist_dir)
            self.logger.info(f"✅ ChromaDB initialized: {persist_dir}")
            
            # Initialize chat history manager
            from senter.chat_history import ChatHistoryManager
            self.chat_history_manager = ChatHistoryManager(self.db, self.user_profile)
            if self.chat_history_manager.initialize():
                self.logger.info("✅ Chat history manager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def _initialize_tts_system(self) -> bool:
        """Initialize text-to-speech system."""
        if not self.config.audio.tts_enabled:
            self.logger.info("🔇 TTS disabled in configuration")
            return True
            
        try:
            from senter.tts_service import TTSService
            
            # Convert config object to dict if needed
            tts_config = {}
            if hasattr(self.config, 'audio'):
                tts_config = {
                    'enabled': getattr(self.config.audio, 'tts_enabled', True),
                    'model_dir': 'piper_models',
                    'model_filename': 'en_US-lessac-medium.onnx'
                }
            
            self.tts_system = TTSService(config=tts_config, user_profile=self.user_profile)
            if self.tts_system.initialize():
                self.logger.info("✅ TTS system initialized")
                return True
            else:
                self.logger.warning("⚠️  TTS initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"TTS initialization failed: {e}")
            return False
    
    def _initialize_attention_detection(self) -> bool:
        """Initialize attention detection system."""
        if not self.config.video.camera_enabled:
            self.logger.info("📷 Camera disabled in configuration")
            return True
            
        try:
            # Setup AvA with callback to our instance method
            from SenterUI.AvA.ava import main as ava_main, set_cli_voice_callback
            
            # Set the voice callback to our handler
            set_cli_voice_callback(self.handle_voice_input)
            
            # Start AvA in a separate thread
            import threading
            self.ava_thread = threading.Thread(target=ava_main, daemon=True)
            self.ava_thread.start()
            
            self.logger.info("✅ Attention detection initialized")
            return True
                
        except Exception as e:
            self.logger.error(f"Attention detection initialization failed: {e}")
            return False
    
    def handle_voice_input(self, user_input: str):
        """Handle voice input from attention detection system."""
        start_time = time.time()
        try:
            self.logger.info(f"Voice input received: {user_input}")
            
            # Log the DetectVoiceCommand action
            self.state_logger.log_action(
                "DetectVoiceCommand",
                "AvA (Attention/Whisper Thread)",
                details={
                    "user_input": user_input,
                    "input_length": len(user_input)
                },
                preconditions={
                    "attention_state": self.state_logger.get_current_state().attention_state.value,
                    "audio_recording_state": self.state_logger.get_current_state().audio_recording_state.value
                }
            )
            
            # Validate input
            if not user_input or not isinstance(user_input, str):
                self.logger.warning("Invalid voice input received")
                self.state_logger.log_action(
                    "DetectVoiceCommand",
                    "AvA (Attention/Whisper Thread)",
                    success=False,
                    error_message="Invalid voice input received"
                )
                return
            
            # Skip empty or very short input
            if len(user_input.strip()) < 3:
                self.logger.debug(f"Input too short, ignoring: '{user_input}'")
                self.state_logger.log_action(
                    "DetectVoiceCommand",
                    "AvA (Attention/Whisper Thread)",
                    success=False,
                    error_message=f"Input too short: '{user_input}'"
                )
                return
            
            # Emergency stop commands
            stop_commands = ["stop", "cancel", "shut up", "quiet", "silence", "enough", "pause"]
            if any(cmd in user_input.lower() for cmd in stop_commands):
                self.logger.info(f"Stop command detected: '{user_input}'")
                self.state_logger.log_action(
                    "ProcessEmergencyStop",
                    "Main Thread",
                    details={"command": user_input},
                    effects={"tts_stopped": True}
                )
                if self.tts_system:
                    self.tts_system.emergency_stop()
                    self.tts_system.speak_sentence("Stopped.")
                return
            
            # Provide immediate acknowledgment
            self.logger.info(f"Processing meaningful command: '{user_input}'")
            if self.tts_system:
                self.tts_system.speak_sentence("Right away!")
            
            # Check for instant lights commands
            instant_result = self._handle_instant_lights(user_input)
            if isinstance(instant_result, dict) and instant_result.get('executed', False):
                self.logger.info("Instant lights command executed")
                return
            
            # Process normally
            self.process_user_input(user_input)
            
            # Log completion
            duration_ms = (time.time() - start_time) * 1000
            self.state_logger.log_action(
                "DetectVoiceCommand",
                "AvA (Attention/Whisper Thread)",
                details={"processed_successfully": True},
                success=True,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error handling voice input: {e}")
            duration_ms = (time.time() - start_time) * 1000
            self.state_logger.log_action(
                "DetectVoiceCommand",
                "AvA (Attention/Whisper Thread)",
                success=False,
                error_message=str(e),
                duration_ms=duration_ms
            )
            # Speak error message if TTS is available
            if self.tts_system:
                try:
                    self.tts_system.speak_text("Sorry, there was an error processing your request.")
                except:
                    pass
    
    def _handle_instant_lights(self, user_input: str) -> dict:
        """Handle instant lights commands for immediate response."""
        import re
        start_time = time.time()
        
        lights_status = {
            'detected': False,
            'executed': False,
            'commands': [],
            'results': []
        }
        
        user_lower = user_input.lower()
        
        # Check for lights keywords first
        if any(word in user_lower for word in ['light', 'lights', 'turn on', 'turn off']):
            lights_status['detected'] = True
            
            # Log the ProcessInstantLights action start
            self.state_logger.log_action(
                "ProcessInstantLights",
                "Main Thread (process_voice_input)",
                details={"user_input": user_input, "step": "detection"},
                preconditions={"lights_keywords_detected": True}
            )
            
            detected_commands = []
            
            # Pattern matching for various light commands
            all_lights_pattern = r'\bturn\s+(?:all\s+)?(?:the\s+)?lights?\s+(red|blue|green|yellow|orange|purple|pink|white|teal|bright|dim|on|off)\b'
            matches = re.findall(all_lights_pattern, user_lower, re.IGNORECASE)
            for match in matches:
                detected_commands.append(f"ALL {match.title()}")
            
            lights_color_pattern = r'\blights?\s+(red|blue|green|yellow|orange|purple|pink|white|teal|bright|dim|on|off)\b'
            matches = re.findall(lights_color_pattern, user_lower, re.IGNORECASE)
            for match in matches:
                detected_commands.append(f"ALL {match.title()}")
            
            color_lights_pattern = r'\b(red|blue|green|yellow|orange|purple|pink|white|teal|bright|dim)\s+lights?\b'
            matches = re.findall(color_lights_pattern, user_lower, re.IGNORECASE)
            for match in matches:
                detected_commands.append(f"ALL {match.title()}")
            
            room_pattern = r'\b(kitchen|living\s+room|bedroom|porch|desk)\s+(?:lights?\s+)?(red|blue|green|yellow|orange|purple|pink|white|teal|bright|dim|on|off)\b'
            matches = re.findall(room_pattern, user_lower, re.IGNORECASE)
            for room, action in matches:
                room_name = "Living Room" if "living" in room.lower() else room.title()
                detected_commands.append(f"{room_name} {action.title()}")
            
            # Simple on/off commands
            if re.search(r'\bturn\s+(?:all\s+)?(?:the\s+)?lights?\s+on\b', user_lower):
                detected_commands.append("ALL ON")
            elif re.search(r'\bturn\s+(?:all\s+)?(?:the\s+)?lights?\s+off\b', user_lower):
                detected_commands.append("ALL OFF")
            elif re.search(r'\bturn\s+on\s+(?:all\s+)?(?:the\s+)?lights?\b', user_lower):
                detected_commands.append("ALL ON")
            elif re.search(r'\bturn\s+off\s+(?:all\s+)?(?:the\s+)?lights?\b', user_lower):
                detected_commands.append("ALL OFF")
            
            # Remove duplicates
            seen = set()
            unique_commands = []
            for cmd in detected_commands:
                if cmd not in seen:
                    seen.add(cmd)
                    unique_commands.append(cmd)
            
            # Execute commands instantly
            if unique_commands:
                self.logger.info(f"🚀 INSTANT LIGHTS: Detected commands: {unique_commands}")
                
                # Update system mode for instant execution
                self.state_logger.update_system_mode(SystemMode.EXECUTING_TOOL, "Instant lights execution")
                
                for command in unique_commands:
                    try:
                        self.logger.debug(f"💡 INSTANT: Executing '{command}'")
                        success = execute_light_command(command)
                        lights_status['commands'].append(command)
                        lights_status['results'].append({
                            'command': command,
                            'actual': command,
                            'success': success
                        })
                        if success:
                            lights_status['executed'] = True
                            self.logger.info(f"✅ INSTANT: '{command}' completed!")
                    except Exception as e:
                        self.logger.error(f"❌ INSTANT: Failed '{command}': {e}")
                
                # Log the ProcessInstantLights action completion
                duration_ms = (time.time() - start_time) * 1000
                self.state_logger.log_action(
                    "ProcessInstantLights",
                    "Main Thread (process_voice_input)",
                    details={
                        "commands": unique_commands,
                        "results": lights_status['results'],
                        "step": "execution_complete"
                    },
                    effects={
                        "system_mode_transition": f"Idle → ExecutingTool → Idle",
                        "commands_executed": len(unique_commands),
                        "successful_commands": sum(1 for r in lights_status['results'] if r['success'])
                    },
                    success=lights_status['executed'],
                    duration_ms=duration_ms
                )
                
                # Return to idle state
                self.state_logger.update_system_mode(SystemMode.IDLE, "Instant lights execution complete")
        
        return lights_status
    
    async def initialize(self) -> bool:
        """Initialize all SENTER components."""
        self.logger.info("🚀 Initializing SENTER system...")
        
        # Clear screen and show UI
        SenterUI.clear_screen()
        SenterUI.show_ascii_logo()
        
        # Initialize components in order
        initialization_steps = [
            ("👤 User Profile System", self._initialize_user_profile),
            ("🧠 AI Models", self._initialize_ai_models), 
            ("🔧 Database", self._initialize_database),
            ("🎤 TTS System", self._initialize_tts_system),
            ("👁️  Attention Detection", self._initialize_attention_detection),
        ]
        
        for step_name, init_func in initialization_steps:
            self.logger.info(f"Initializing {step_name}...")
            print(f"🔄 {step_name}...")
            
            try:
                if not init_func():
                    self.logger.error(f"Failed to initialize {step_name}")
                    return False
                    
                print(f"✅ {step_name} initialized")
                await asyncio.sleep(0.1)  # Brief pause for UI
                
            except Exception as e:
                self.logger.error(f"Error initializing {step_name}: {e}")
                return False
        
        # Initialize journal system
        try:
            initialize_journal_system(self.db, self.user_profile)
            self.logger.info("✅ Journal system initialized")
        except Exception as e:
            self.logger.warning(f"Journal system initialization failed: {e}")
        
        self.logger.info("🎯 SENTER initialization complete!")
        return True
    
    def _generate_ai_response(self, user_input: str) -> str:
        """Generate AI response using tools model."""
        try:
            # Build dynamic system prompt
            system_prompt = self._build_system_prompt(user_input)
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'PROMPT: {user_input}'}
            ]
            
            return self._generate_response_with_validation(messages)
            
        except Exception as e:
            self.logger.error(f"Error generating AI response: {e}")
            return ""
    
    def _build_system_prompt(self, user_input: str) -> str:
        """Build dynamic system prompt with relevant context."""
        relevant_tools = self._determine_relevant_tools(user_input)
        
        prompt = '''You are Senter. ALWAYS use tools for information requests.

AVAILABLE TOOLS:
<announcement>brief response</announcement>
<lights>ALL ON/OFF/Red/Blue/Green/etc</lights>
<research>search query</research>
<camera>front camera</camera>
<journal>search past conversations</journal>

CRITICAL RULES:
- For "tell me about/story about [topic]" → ALWAYS use <research>topic</research>
- NEVER generate fictional content directly - use research tool first
- Multiple tools allowed: <announcement>text</announcement><lights>command</lights><research>query</research>
- NO nesting tools inside each other

EXAMPLES:
"turn lights red and tell me about UFOs" → <announcement>Turning lights red and researching UFOs</announcement><lights>ALL Red</lights><research>UFO sightings encounters</research>
"how do I look" → <announcement>Let me take a look</announcement><camera>front camera</camera>
"what did we discuss about aliens" → <announcement>Checking our conversation history</announcement><journal>alien discussion</journal>
'''
        
        # Add relevant chat history
        if self.chat_history_manager:
            try:
                history = self.chat_history_manager.get_relevant_history(user_input)
                if history:
                    formatted_history = self.chat_history_manager.format_history_for_prompt(history)
                    prompt += formatted_history
            except Exception as e:
                self.logger.warning(f"Error adding chat history: {e}")
        
        return prompt
    
    def _execute_tools(self, response_text: str, user_input: str) -> bool:
        """Execute tool commands from AI response."""
        import re
        tools_executed = False
        
        try:
            # Handle lights commands
            lights_commands = re.findall(r'<lights>\s*([^<]*?)\s*</lights>', response_text, re.DOTALL | re.IGNORECASE)
            if lights_commands:
                self.logger.info(f"🔧 Found {len(lights_commands)} lights command(s)")
                success_count = 0
                
                for i, lights_command in enumerate(lights_commands):
                    lights_command = lights_command.strip()
                    if lights_command:
                        self.logger.debug(f"🔧 Executing lights command {i+1}: {lights_command}")
                        success = execute_light_command(lights_command)
                        if success:
                            self.logger.info(f"✅ Command completed: {lights_command}")
                            success_count += 1
                        else:
                            self.logger.warning(f"❌ Command failed: {lights_command}")
                
                if success_count > 0:
                    self.logger.info(f"✅ {success_count} lights command(s) completed!")
                    tools_executed = True
            
            # Handle research commands
            research_commands = re.findall(r'<research>\s*([^<]*?)\s*</research>', response_text, re.DOTALL | re.IGNORECASE)
            if research_commands:
                self.logger.info(f"🔍 Found {len(research_commands)} research command(s)")
                
                for i, research_command in enumerate(research_commands):
                    research_command = research_command.strip()
                    if research_command:
                        self.logger.debug(f"🔍 Executing research command {i+1}: {research_command}")
                        
                        try:
                            # Create TTS callback if available
                            tts_callback = self.tts_system.speak_sentence if self.tts_system else None
                            research_results = execute_research(research_command, tts_callback)
                            self.logger.info(f"✅ Research {i+1} completed!")
                            
                            # Generate AI response for the last research
                            if i == len(research_commands) - 1 and len(research_results) > 100:
                                ai_response = self._generate_ai_response_from_research(
                                    user_input, research_results, tts_callback, research_command
                                )
                                self.logger.debug(f"AI Response generated from research")
                            
                            tools_executed = True
                        except Exception as e:
                            self.logger.error(f"❌ Research {i+1} failed: {e}")
            
            # Handle camera commands
            camera_commands = re.findall(r'<camera>\s*([^<]*?)\s*</camera>', response_text, re.DOTALL | re.IGNORECASE)
            if camera_commands and CAMERA_TOOLS_AVAILABLE:
                self.logger.info(f"📸 Found {len(camera_commands)} camera command(s)")
                
                for i, camera_command in enumerate(camera_commands):
                    camera_command = camera_command.strip()
                    if camera_command:
                        self.logger.debug(f"📸 Executing camera command {i+1}: {camera_command}")
                        
                        try:
                            tts_callback = self.tts_system.speak_sentence if self.tts_system else None
                            success = execute_camera_command(camera_command, tts_callback, self.attention_detector)
                            
                            if success:
                                self.logger.info(f"✅ Camera command {i+1} completed!")
                                tools_executed = True
                            else:
                                self.logger.warning(f"❌ Camera command {i+1} failed")
                        except Exception as e:
                            self.logger.error(f"❌ Camera command {i+1} error: {e}")
            
            # Handle journal commands
            journal_commands = re.findall(r'<journal>\s*([^<]*?)\s*</journal>', response_text, re.DOTALL | re.IGNORECASE)
            if journal_commands:
                self.logger.info(f"📖 Found {len(journal_commands)} journal command(s)")
                
                for i, journal_command in enumerate(journal_commands):
                    journal_command = journal_command.strip()
                    if journal_command:
                        self.logger.debug(f"📖 Executing journal search {i+1}: {journal_command}")
                        
                        try:
                            journal_results = self._search_journal(journal_command)
                            self.logger.info(f"✅ Journal search {i+1} completed!")
                            
                            # Generate AI response for the last search
                            if i == len(journal_commands) - 1 and len(journal_results) > 100:
                                tts_callback = self.tts_system.speak_sentence if self.tts_system else None
                                ai_response = self._generate_ai_response_from_research(
                                    user_input, journal_results, tts_callback, journal_command
                                )
                                self.logger.debug(f"AI Response generated from journal")
                            
                            tools_executed = True
                        except Exception as e:
                            self.logger.error(f"❌ Journal search {i+1} failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Error executing tools: {e}")
        
        return tools_executed
    
    def _extract_announcement(self, response_text: str) -> str:
        """Extract announcement text from AI response."""
        import re
        announcement_match = re.search(r'<announcement>\s*([^<]*?)\s*</announcement>', response_text, re.DOTALL | re.IGNORECASE)
        if announcement_match:
            text = announcement_match.group(1).strip()
            text = re.sub(r'<[^>]*>', '', text).strip()
            text = ' '.join(text.split())
            return text
        return ""

    def _generate_response_with_validation(self, messages: list, max_retries: int = 2) -> str:
        """Generate response with validation and retry mechanism."""
        import re
        
        valid_tools = ['research', 'lights', 'camera', 'journal']
        
        for attempt in range(max_retries):
            try:
                t_start = time.time()
                
                # Adjust parameters for retries
                if attempt == 0:
                    temperature = 0.0
                    top_k = 1
                    top_p = 0.1
                    max_tokens = 50
                else:
                    temperature = 0.1
                    top_k = 3
                    top_p = 0.3
                    max_tokens = 75
                
                if attempt > 0:
                    self.logger.debug(f"🔄 Retry {attempt}/{max_retries - 1} with temp={temperature}")
                
                response = self.senter_tools.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=1.02,
                    stop=["Human:", "User:", "\n\nHuman:", "\n\nUser:", "</tool>", "```", "\n\n"]
                )
                
                t_end = time.time()
                full_response = response['choices'][0]['message']['content']
                
                token_count = len(full_response.split())
                tokens_per_second = token_count / (t_end - t_start) if (t_end - t_start) > 0 else 0
                
                self.logger.debug(f"⚡ Tool generation: {t_end - t_start:.2f}s, {token_count} tokens ({tokens_per_second:.1f} tok/s)")
                
                # Clean and validate response
                cleaned_response = self._clean_malformed_xml(full_response)
                
                # Validation
                has_announcement = '<announcement>' in cleaned_response and '</announcement>' in cleaned_response
                found_tools = []
                hallucinated_tools = []
                
                all_tags = re.findall(r'<(\w+)>', cleaned_response)
                for tag in set(all_tags):
                    if tag == 'announcement':
                        continue
                    if tag in valid_tools:
                        found_tools.append(tag)
                    else:
                        hallucinated_tools.append(tag)
                
                if has_announcement and found_tools and not hallucinated_tools:
                    self.logger.debug(f"✅ Valid response with tools: {found_tools}")
                    return cleaned_response
                elif hallucinated_tools and attempt < max_retries - 1:
                    self.logger.warning(f"❌ Hallucinated tools detected: {hallucinated_tools}, retrying...")
                    continue
                elif attempt == max_retries - 1:
                    # Create fallback response on final attempt
                    if hallucinated_tools and not found_tools:
                        user_query = "your request"
                        if messages and len(messages) > 1:
                            user_content = messages[-1].get('content', '')
                            if 'PROMPT:' in user_content:
                                user_query = user_content.split('PROMPT:')[-1].strip()
                        return f"<announcement>Let me research that for you</announcement><research>{user_query}</research>"
                    return cleaned_response
                    
            except Exception as e:
                self.logger.error(f"❌ Generation error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return f"<announcement>Error occurred</announcement><research>help with request</research>"
        
        return f"<announcement>Unable to process request</announcement><research>general help</research>"

    def _clean_malformed_xml(self, response_text: str) -> str:
        """Clean up common XML formatting issues."""
        import re
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', response_text.strip())
        
        # Fix broken XML tags
        cleaned = re.sub(r'</(\w+)\s*\n\s*>', r'</\1>', cleaned)
        cleaned = re.sub(r'<(\w+)>\s*\n\s*([^<]*?)\s*\n\s*</\1>', r'<\1>\2</\1>', cleaned)
        cleaned = re.sub(r'</(\w+)[^>]*>', r'</\1>', cleaned)
        cleaned = re.sub(r'<(\w+)([^>]*)\n\s*>', r'<\1\2>', cleaned)
        
        return cleaned.strip()

    def _determine_relevant_tools(self, query: str) -> list:
        """Determine which tools are relevant to the query."""
        relevant_tools = set()
        query_lower = query.lower()
        
        # Camera keywords (prioritize for appearance questions)
        camera_keywords = [
            'photo', 'picture', 'camera', 'take', 'capture', 'snap', 'selfie',
            'how do i look', 'how does my hair look', 'how does my', 'do i look',
            'screenshot', 'screen shot', 'what\'s on my screen'
        ]
        
        appearance_keywords = [
            'how do i look', 'how i look', 'how does my hair look', 'do i look',
            'my appearance', 'look good', 'look bad', 'hair looks'
        ]
        
        is_appearance_question = any(keyword in query_lower for keyword in appearance_keywords)
        
        if any(keyword in query_lower for keyword in camera_keywords):
            relevant_tools.add('camera')
        
        # Light keywords
        light_keywords = ['light', 'lights', 'turn on', 'turn off', 'red', 'blue', 'green', 'color', 'bright', 'dim']
        if any(keyword in query_lower for keyword in light_keywords):
            relevant_tools.add('lights')
        
        # Journal keywords
        journal_keywords = [
            'what did we', 'what were we', 'did we discuss', 'were we talking',
            'conversation about', 'remember when', 'recall our', 'our previous'
        ]
        if any(keyword in query_lower for keyword in journal_keywords):
            relevant_tools.add('journal')
        
        # Research for questions (but not appearance or memory questions)
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'tell me', 'about', '?']
        is_question = any(word in query_lower for word in question_words)
        is_memory_question = any(keyword in query_lower for keyword in journal_keywords)
        
        if is_question and not is_appearance_question and not is_memory_question:
            relevant_tools.add('research')
        
        # Default to research if no tools detected
        if not relevant_tools:
            relevant_tools.add('research')
        
        return list(relevant_tools)

    def _search_journal(self, query: str) -> str:
        """Search through journal entries and chat history."""
        results = []
        
        # Search chat history for relevant conversations
        if self.chat_history_manager:
            try:
                relevant_history = self.chat_history_manager.get_relevant_history(query, max_results=5)
                
                if relevant_history:
                    results.append("🗨️ **RELEVANT CHAT HISTORY:**")
                    results.append("=" * 40)
                    
                    for i, exchange in enumerate(relevant_history):
                        similarity_info = f" (similarity: {exchange['similarity']:.2f})" if exchange['similarity'] > 0 else " (recent)"
                        timestamp = datetime.fromtimestamp(exchange['timestamp']).strftime("%Y-%m-%d %H:%M")
                        
                        results.append(f"\n📅 **Exchange {i+1}** - {timestamp}{similarity_info}")
                        results.append(exchange['exchange'])
                        results.append("-" * 30)
                
            except Exception as e:
                self.logger.warning(f"Error searching chat history: {e}")
        
        # Search journal system for session data
        try:
            from journal_system import journal_system
            
            if journal_system and hasattr(journal_system, 'journal_collection') and journal_system.journal_collection:
                journal_results = journal_system.journal_collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas"]
                )
                
                if journal_results and journal_results['documents'] and journal_results['documents'][0]:
                    if results:
                        results.append("\n\n")
                    
                    results.append("📝 **JOURNAL SESSIONS:**")
                    results.append("=" * 40)
                    
                    for i, (doc, metadata) in enumerate(zip(journal_results['documents'][0], journal_results['metadatas'][0])):
                        timestamp = datetime.fromtimestamp(metadata['timestamp']).strftime("%Y-%m-%d %H:%M")
                        interaction_count = metadata.get('interaction_count', 0)
                        
                        results.append(f"\n📅 **Session {i+1}** - {timestamp} ({interaction_count} interactions)")
                        
                        try:
                            import json
                            session_data = json.loads(doc)
                            interactions = session_data.get('interactions', [])
                            topics = session_data.get('topics_discussed', [])
                            
                            if topics:
                                results.append(f"🏷️  Topics discussed: {', '.join(topics)}")
                            
                            for interaction in interactions[:3]:
                                results.append(f"User: {interaction['user_input']}")
                                results.append(f"Assistant: {interaction['ai_response']}")
                                if interaction.get('tools_used'):
                                    results.append(f"Tools used: {', '.join(interaction['tools_used'])}")
                                results.append("")
                        
                        except Exception as e:
                            results.append(f"Session data: {doc[:200]}...")
                        
                        results.append("-" * 30)
        
        except Exception as e:
            self.logger.warning(f"Error searching journal sessions: {e}")
        
        if not results:
            return f"📖 No relevant information found in chat history or journal for: '{query}'"
        
        final_result = "\n".join(results)
        if len(final_result) > 2000:
            final_result = final_result[:2000] + "\n\n... [Results truncated for brevity]"
        
        return final_result

    def _generate_ai_response_from_research(self, original_question: str, research_results: str, 
                                          tts_callback=None, search_query: str = None) -> str:
        """Generate AI response using research results with streaming."""
        
        # Brief delay for coordination
        time.sleep(0.1)
        
        # Generate thinking announcement
        thinking_query = search_query if search_query else original_question
        thinking_sentences = self._generate_thinking_announcement(thinking_query)
        
        full_thinking_text = " ".join(thinking_sentences)
        self.logger.info(f"🤔 {full_thinking_text}")
        
        if tts_callback:
            for sentence in thinking_sentences:
                tts_callback(sentence)
            time.sleep(0.1)
        
        # Filter content for relevance
        filtered_content = self._filter_most_relevant_content(research_results, original_question)
        
        rn = datetime.now()
        prompt = f"""Based on the research results below, please answer this question: {original_question}

Research Results:
{filtered_content[:1500]}

Please provide a clear, concise answer focusing on the most important information."""

        try:
            messages = [{
                'role': 'system',
                'content': f'''Cutting Knowledge Date: December 2023 
                            Today Date and Time: {rn}
                            
                            You are a helpful assistant. Answer the user's question clearly and thoroughly based on the research results provided.
                            Provide a comprehensive response with good detail and context. Aim for 250-400 words when appropriate.'''
            }, {
                'role': 'user',
                'content': prompt
            }]
            
            # Stream the response
            sentence_buffer = ""
            first_sentence_queued = False
            full_response_text = ""
            token_count = 0
            first_token_time = None
            
            self.logger.debug("🎯 Streaming AI response...")
            t_start = time.time()
            
            stream = self.senter_response.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                stream=True,
                top_p=0.9,
                top_k=20,
                repeat_penalty=1.0,
                stop=["Human:", "User:", "\n\nHuman:", "\n\nUser:"]
            )
            
            for chunk in stream:
                chunk_text = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                if chunk_text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - t_start
                        self.logger.debug(f"🚀 First token: {ttft:.2f}s")
                    
                    token_count += len(chunk_text.split())
                    full_response_text += chunk_text
                    sentence_buffer += chunk_text

                    # Simple sentence detection
                    while True:
                        import re
                        match = re.search(r"([.?!])", sentence_buffer)
                        if match:
                            end_index = match.end()
                            sentence = sentence_buffer[:end_index].strip()
                            if sentence:
                                if not first_sentence_queued:
                                    first_sentence_queued = True
                                    # Clear TTS queue to prevent overlap
                                    if self.tts_system:
                                        self.tts_system.emergency_stop()
                                
                                if tts_callback:
                                    tts_callback(sentence)

                            sentence_buffer = sentence_buffer[end_index:].lstrip()
                        else:
                            break

            # Queue remaining text
            if sentence_buffer.strip() and tts_callback:
                tts_callback(sentence_buffer.strip())

            t_end = time.time()
            total_time = t_end - t_start
            tokens_per_second = token_count / total_time if total_time > 0 else 0
            self.logger.debug(f"✅ Streaming completed: {total_time:.2f}s, {token_count} tokens ({tokens_per_second:.1f} tok/s)")
            
            return full_response_text if len(full_response_text) > 10 else f"Based on the research: {filtered_content[:300]}..."
            
        except Exception as e:
            self.logger.error(f"Error generating AI response: {e}")
            return f"Here's what I found: {filtered_content[:400]}..."

    def _generate_thinking_announcement(self, query: str) -> list:
        """Generate a thinking announcement as sentences."""
        import random
        
        thinking_phrase_sets = [
            [f"Let me think about {query}."],
            [f"Give me a moment.", f"I'll look into {query}."],
            [f"Let me consider {query}."],
            [f"Let me think about this.", f"I'll research {query} for you."],
            [f"I'm reading up on {query}.", "Give me a moment."],
            [f"Ok, let me get my thoughts together.", f"I'll find information about {query}."],
            [f"Let me see what I can find.", f"Researching {query} now."],
            [f"Interesting question about {query}.", "Let me think about that."],
            [f"Alright, let me research {query}.", "This should be helpful."],
            [f"Ok, {query}.", "Let me look into that for you."],
        ]
        
        return random.choice(thinking_phrase_sets)

    def _filter_most_relevant_content(self, research_results: str, original_question: str) -> str:
        """Filter research results to extract most relevant content."""
        import re
        
        # Split research results into source blocks
        source_blocks = research_results.split("📖 SOURCE")
        
        if len(source_blocks) < 2:
            return research_results[:3000] + "..." if len(research_results) > 3000 else research_results
        
        # Score sources for relevance
        scored_sources = []
        question_keywords = set(original_question.lower().split())
        
        for i, block in enumerate(source_blocks[1:], 1):
            if not block.strip():
                continue
                
            content_match = re.search(r'📄 \*\*Content:\*\* (.+?)(?=\n─|$)', block, re.DOTALL)
            if not content_match:
                continue
                
            content = content_match.group(1).strip()
            content_words = set(content.lower().split())
            keyword_overlap = len(question_keywords.intersection(content_words))
            length_score = min(len(content) / 200, 2.0)
            relevance_score = keyword_overlap + length_score
            
            scored_sources.append((relevance_score, f"📖 SOURCE {i}: {block}", len(content)))
        
        # Sort by relevance and select top sources
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        
        selected_sources = []
        total_length = 0
        max_length = 2500
        
        for score, source_block, content_length in scored_sources:
            if total_length + content_length > max_length and selected_sources:
                break
            selected_sources.append(source_block)
            total_length += content_length
            
            if len(selected_sources) >= 3:
                break
                
        if selected_sources:
            header_match = re.search(r'^(🔍 \*\*RESEARCH RESULTS FOR:\*\*.*?\n=+\n\n)', research_results, re.DOTALL)
            header = header_match.group(1) if header_match else "🔍 **FILTERED RESEARCH RESULTS:**\n" + "="*40 + "\n\n"
            
            filtered_content = header + "\n\n".join(selected_sources)
            filtered_content += f"\n\n**📋 Note:** Showing top {len(selected_sources)} most relevant sources"
            
            return filtered_content
        else:
            return research_results[:2500] + "..." if len(research_results) > 2500 else research_results

    def process_user_input(self, user_input: str) -> bool:
        """Process user input and execute appropriate actions."""
        if not user_input or not user_input.strip():
            return True
            
        self.logger.info(f"Processing: {user_input}")
        start_time = time.time()
        
        try:
            # Log ProcessLLMRequest action start
            self.state_logger.update_system_mode(SystemMode.PROCESSING, "Processing user input")
            self.state_logger.log_action(
                "ProcessLLMRequest",
                "Main Thread (process_voice_input)",
                details={
                    "user_input": user_input,
                    "step": "start_processing"
                },
                preconditions={
                    "system_mode": "Processing",
                    "instant_lights_skipped": True
                }
            )
            
            # Generate AI response
            response_text = self._generate_ai_response(user_input)
            
            if response_text:
                # Extract announcement for TTS
                announcement = self._extract_announcement(response_text)
                if announcement and self.tts_system:
                    self.tts_system.speak_text(announcement)
                
                # Update system mode for tool execution
                self.state_logger.update_system_mode(SystemMode.EXECUTING_TOOL, "Executing tools from LLM response")
                
                # Execute tool commands
                tools_executed = self._execute_tools(response_text, user_input)
                
                # Update ChromaDB state
                self.state_logger.log_action(
                    "UpdateChromaDBState",
                    "Main Thread",
                    details={
                        "user_input": user_input,
                        "ai_response_length": len(response_text),
                        "tools_executed": tools_executed
                    },
                    effects={"database_updated": True}
                )
                
                # Track interaction in journal
                add_interaction_to_journal(
                    user_input=user_input,
                    ai_response=response_text,
                    tools_used=None,
                    tool_results="Tools executed" if tools_executed else None
                )
                
                # Save to chat history
                if self.chat_history_manager:
                    self.chat_history_manager.save_exchange(
                        user_prompt=user_input,
                        ai_response=response_text,
                        tool_results="Tools executed" if tools_executed else None
                    )
                
                # Return to idle state
                self.state_logger.update_system_mode(SystemMode.IDLE, "Processing complete")
                
                # Log ProcessLLMRequest completion
                duration_ms = (time.time() - start_time) * 1000
                self.state_logger.log_action(
                    "ProcessLLMRequest",
                    "Main Thread (process_voice_input)",
                    details={
                        "response_length": len(response_text),
                        "tools_executed": tools_executed,
                        "step": "complete"
                    },
                    effects={
                        "system_mode_transitions": "Idle → Processing → ExecutingTool → Idle",
                        "database_updated": True,
                        "announcement_queued": bool(announcement)
                    },
                    success=True,
                    duration_ms=duration_ms
                )
                
                return True
            else:
                self.logger.error("No response generated")
                self.state_logger.update_system_mode(SystemMode.IDLE, "Processing failed - no response")
                self.state_logger.log_action(
                    "ProcessLLMRequest",
                    "Main Thread (process_voice_input)",
                    success=False,
                    error_message="No response generated",
                    duration_ms=(time.time() - start_time) * 1000
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            self.state_logger.update_system_mode(SystemMode.IDLE, "Processing failed with error")
            self.state_logger.log_action(
                "ProcessLLMRequest",
                "Main Thread (process_voice_input)",
                success=False,
                error_message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            return False
    
    async def run_interactive_mode(self):
        """Run interactive text mode."""
        self.logger.info("💬 Starting interactive mode")
        print(f"\n💬 Text input mode - type commands or 'exit' to quit")
        print(f"   User: {self.user_profile.get_display_name()}")
        
        while not self.shutdown_event.is_set():
            try:
                prompt = input(f"\n{self.user_profile.get_display_name().upper()}: ")
                
                # Handle exit commands
                if prompt.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("👋 Goodbye!")
                    break
                
                # Process input
                self.process_user_input(prompt)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
    
    async def run_attention_mode(self):
        """Run with attention detection (voice/camera input)."""
        self.logger.info("👁️  Starting attention mode")
        print(f"\n👁️  Attention mode - look at camera or speak commands")
        print(f"   Use Ctrl+C to exit")
        
        try:
            # Keep running until shutdown
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    
    async def shutdown(self):
        """Gracefully shutdown the application."""
        self.logger.info("🛑 Shutting down SENTER...")
        self.shutdown_event.set()
        
        # Log shutdown action
        self.state_logger.log_action(
            "SystemShutdown",
            "SenterApplication",
            details={"shutdown_initiated": True}
        )
        
        # Emergency stop TTS first
        if self.tts_system:
            try:
                self.tts_system.emergency_stop()
            except Exception as e:
                self.logger.error(f"Error stopping TTS: {e}")
        
        # Stop attention detection/AvA
        try:
            if hasattr(self, 'ava_thread') and self.ava_thread.is_alive():
                # Try to stop AvA gracefully
                from SenterUI.AvA.ava import stop_ava
                stop_ava()
                self.ava_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.warning(f"Error stopping attention detection: {e}")
        
        # Shutdown TTS system properly
        if self.tts_system:
            try:
                self.tts_system.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down TTS system: {e}")
        
        # Clean up audio system
        try:
            import sounddevice as sd
            sd.stop()
            sd.default.reset()
        except Exception as e:
            self.logger.warning(f"Error cleaning up audio: {e}")
        
        # Clean up process manager if available
        try:
            if OPTIMIZATION_AVAILABLE:
                from process_manager import stop_monitoring
                stop_monitoring()
        except Exception as e:
            self.logger.warning(f"Error stopping process manager: {e}")
        
        # Close state logger and save summary
        try:
            self.state_logger.log_action(
                "SystemShutdown",
                "SenterApplication",
                details={"shutdown_complete": True},
                success=True
            )
            self.state_logger.close()
            self.logger.info("✅ State logger closed and summary saved")
        except Exception as e:
            self.logger.error(f"Error closing state logger: {e}")
        
        self.logger.info("✅ SENTER shutdown complete")

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\n📡 Received signal {signum}")
    # Will be handled by the main loop
    
async def main():
    """Main application entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and initialize application
    app = SenterApplication()
    
    try:
        # Initialize all systems
        if not await app.initialize():
            print("❌ SENTER initialization failed")
            return 1
        
        # Show welcome message
        config = get_config()
        SenterUI.show_welcome_message(
            app.user_profile.get_display_name(),
            app.user_profile.get_greeting_style()
        )
        
        # Run in appropriate mode
        if config.video.camera_enabled:
            await app.run_attention_mode()
        else:
            await app.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return 1
    finally:
        await app.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 


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
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'system_mode': self.system_mode.value,
            'attention_state': self.attention_state.value,
            'audio_recording_state': self.audio_recording_state.value,
            'tts_queue_size': self.tts_queue_size,
            'active_tts_count': self.active_tts_count,
            'tool_execution_status': self.tool_execution_status,
            'current_user': self.current_user,
            'session_id': self.session_id
        }

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
            'duration_ms': self.duration_ms
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
    """Comprehensive state logging system for SENTER."""
    
    def __init__(self, logs_dir: Path = Path("logs"), session_id: Optional[str] = None):
        """Initialize the state logger."""
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        
        # Generate session ID
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Current state tracking
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
            session_id=self.session_id
        )
        
        # Event storage
        self._actions: List[ActionEvent] = []
        self._state_history: List[StateSnapshot] = []
        self._invariant_violations: List[InvariantViolation] = []
        
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
            duration_ms=duration_ms
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

def initialize_state_logger(logs_dir: Path = Path("logs"), session_id: Optional[str] = None) -> StateLogger:
    """Initialize global state logger."""
    global _state_logger
    _state_logger = StateLogger(logs_dir, session_id)
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

