# chatbot_system.py
"""
Robust Chatbot System with Multi-turn Memory, Command Handling, and API Integration
"""
import json
import logging
import os
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management with environment variable overrides"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from file with environment variable override"""
        try:
            with open("config.json") as f:
                self.config = json.load(f)
            
            # Override with environment variables
            for key in self.config:
                if env_val := os.getenv(key.upper()):
                    self.config[key] = env_val
                    logger.info(f"Overridden {key} from environment")
                    
        except FileNotFoundError:
            logger.warning("config.json not found, using environment variables")
            self.config = {}
            self._load_from_env()
    
    def _load_from_env(self):
        """Load essential parameters from environment variables"""
        essential_keys = ['API_KEY', 'MODEL']
        for key in essential_keys:
            if env_val := os.getenv(key):
                self.config[key.lower()] = env_val
            else:
                logger.error(f"Missing required config: {key}")
                raise ValueError(f"Required config {key} not found")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)

class ChatHistory:
    """Manages conversation context with automatic summarization"""
    def __init__(self, system_prompt: str, max_length: int = 20):
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_length = max_length
        self.summary = ""
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_length:
            self._summarize()
    
    def _summarize(self):
        """Condense conversation history when it grows too long"""
        # This would call an AI summarization service in production
        # For demo, we'll just keep the first system message + last half
        keep_count = self.max_length // 2
        self.summary = f"Conversation up to {datetime.now()}"
        self.history = [self.history[0]] + self.history[-keep_count:]
        logger.info("Conversation history summarized")

    def get_context(self) -> List[Dict]:
        """Get full conversation context with summary"""
        if self.summary:
            return [{"role": "system", "content": self.summary}] + self.history
        return self.history

class CommandHandler:
    """Detects and processes special commands with plugin architecture"""
    def __init__(self):
        self.commands = self._load_commands()
    
    def _load_commands(self) -> Dict:
        """Initialize core commands with plugin support"""
        return {
            "/weather": self._handle_weather,
            "/help": self._handle_help,
            "/config": self._handle_config,
            "/time": self._handle_time,
            # Add more commands here
        }
    
    def register_command(self, command: str, handler):
        """Plugin method to add new commands"""
        self.commands[command] = handler
        logger.info(f"Registered new command: {command}")
    
    def process(self, user_input: str) -> Optional[str]:
        """Check if input matches any command"""
        cmd_parts = user_input.strip().split(maxsplit=1)
        if not cmd_parts:
            return None
            
        cmd = cmd_parts[0].lower()
        if handler := self.commands.get(cmd):
            args = cmd_parts[1] if len(cmd_parts) > 1 else ""
            return handler(args)
        return None
    
    def _handle_weather(self, args: str) -> str:
        """Fetch weather from external API"""
        try:
            config = ConfigManager()
            location = args or config.get("default_location", "New York")
            api_key = config.get("weather_api_key")
            
            if not api_key:
                return "Weather service not configured"
                
            url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return (
                f"Weather in {location}: "
                f"{data['current']['temp_c']}°C, "
                f"{data['current']['condition']['text']}"
            )
        except Exception as e:
            logger.error(f"Weather API error: {str(e)}")
            return "Sorry, I couldn't fetch weather data."
    
    def _handle_help(self, _) -> str:
        """Show available commands"""
        return "Available commands:\n" + "\n".join(
            f"- {cmd}" for cmd in self.commands.keys()
        )
    
    def _handle_config(self, _) -> str:
        """Show current configuration"""
        config = ConfigManager()
        return (
            f"Model: {config.get('model')}\n"
            f"Temperature: {config.get('temperature', 0.7)}"
        )
    
    def _handle_time(self, _) -> str:
        """Return current time"""
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

class AIClient:
    """Handles communication with AI service API"""
    def __init__(self):
        self.config = ConfigManager()
        self.base_url = self.config.get("api_endpoint", "https://api.openai.com/v1/chat/completions")
        self.headers = {
            "Authorization": f"Bearer {self.config.get('api_key')}",
            "Content-Type": "application/json"
        }
    
    def get_response(self, messages: List[Dict]) -> Tuple[str, int]:
        """Get AI response with comprehensive error handling"""
        try:
            payload = {
                "model": self.config.get("model", "gpt-3.5-turbo"),
                "messages": messages,
                "temperature": float(self.config.get("temperature", 0.7)),
                "max_tokens": int(self.config.get("max_tokens", 500))
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=int(self.config.get("timeout", 10))
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"], response.status_code
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection error: {str(e)}")
            return "I'm having trouble connecting to the AI service.", 503
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "Something went wrong with my response generation.", 500

class ChatBot:
    """Core chatbot orchestrator"""
    def __init__(self):
        self.config = ConfigManager()
        self.history = ChatHistory(
            system_prompt=self.config.get("system_prompt", "You're a helpful assistant"),
            max_length=int(self.config.get("max_history_length", 15))
        self.command_handler = CommandHandler()
        self.ai_client = AIClient()
        
        # Register plugins
        self._register_plugins()
    
    def _register_plugins(self):
        """Initialize optional plugins"""
        try:
            # Example plugin registration
            # from plugins.calculator import handle_calc
            # self.command_handler.register_command("/calc", handle_calc)
            pass
        except ImportError:
            logger.warning("Plugin module not available")
    
    def process_message(self, user_input: str) -> Tuple[str, int]:
        """Process user input through full pipeline"""
        # First check for commands
        if command_result := self.command_handler.process(user_input):
            return command_result, 200
        
        # Add user message to history
        self.history.add_message("user", user_input)
        
        # Get AI response
        context = self.history.get_context()
        ai_response, status = self.ai_client.get_response(context)
        
        # Add AI response to history
        if status == 200:
            self.history.add_message("assistant", ai_response)
        
        return ai_response, status

# FastAPI Web Interface
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Enhanced Chatbot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Global chatbot instance (for demo - use proper DI in production)
bot = ChatBot()

@app.post("/chat", summary="Process chat message")
async def chat_endpoint(request: ChatRequest):
    """Main chatbot processing endpoint"""
    response, status_code = bot.process_message(request.message)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response)
    return {"response": response}

@app.get("/commands", summary="List available commands")
async def list_commands():
    """Endpoint to list registered commands"""
    return {"commands": list(bot.command_handler.commands.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
