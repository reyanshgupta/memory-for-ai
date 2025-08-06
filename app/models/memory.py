"""Memory data models using Pydantic v2."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class MemoryType(str, Enum):
    """Types of memory content."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    CODE = "code"


class MemorySource(str, Enum):
    """Sources where memories originate from."""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    CURSOR = "cursor"
    BROWSER = "browser"
    API = "api"
    MANUAL = "manual"


class Memory(BaseModel):
    """Core memory model with content, metadata, and embeddings."""
    
    id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="Memory content")
    type: MemoryType = Field(default=MemoryType.TEXT, description="Type of memory content")
    source: MemorySource = Field(..., description="Source of the memory")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    importance: float = Field(default=1.0, ge=0.0, le=10.0, description="Importance score")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MemoryCollection(BaseModel):
    """Collection of related memories."""
    
    id: str = Field(..., description="Unique collection identifier")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    memories: List[str] = Field(default_factory=list, description="Memory IDs in this collection")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }