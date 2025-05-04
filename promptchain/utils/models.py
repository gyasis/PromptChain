from pydantic import BaseModel, validator
from enum import Enum
from typing import Union, Callable, List, Literal

class ChainStep(BaseModel):
    step: int
    input: str
    output: str
    type: Literal["initial", "model", "function", "mcp_tool"]

class ChainTechnique(str, Enum):
    NORMAL = "normal"
    HEAD_TAIL = "head-tail"

class ChainInstruction(BaseModel):
    instructions: Union[List[Union[str, Callable[[str], str]]], str]
    technique: ChainTechnique

    @validator('instructions')
    def validate_instructions(cls, v, values):
        technique = values.get('technique')
        # Allow validator to pass if technique is not yet available (should not happen with proper ordering)
        if technique is None: 
            return v 
            
        if technique == ChainTechnique.NORMAL:
            if not isinstance(v, list):
                raise ValueError("Normal technique requires a list of instructions")
            if not v: # Check if list is empty
                raise ValueError("Instruction list cannot be empty for NORMAL technique")
            for instruction in v:
                if not isinstance(instruction, (str, Callable)):
                    raise ValueError("Instructions must be strings or callable functions")
        # Add validation for HEAD_TAIL if specific constraints apply
        # elif technique == ChainTechnique.HEAD_TAIL:
        #     if not isinstance(v, str): # Example constraint
        #         raise ValueError("Head-Tail technique currently requires a single string instruction template")
        return v 