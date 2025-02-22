from litellm import completion
import os
from typing import Union, Callable, List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Update the path for .env loading
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) 