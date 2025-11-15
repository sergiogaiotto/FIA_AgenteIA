"""
Configuração simplificada da aplicação
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


class Settings(BaseModel):
    """Configurações da aplicação"""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.1)
    openai_max_tokens: int = Field(default=2000)
    
    # Firecrawl Configuration
    firecrawl_api_key: Optional[str] = Field(default=os.getenv("FIRECRAWL_API_KEY"))
    firecrawl_timeout: int = Field(default=30)
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(default=os.getenv("PINECONE_API_KEY"))
    pinecone_index_name: str = Field(default="fia-agente-ia")
    
    # Flowise Configuration
    api_externo_agent: str = Field(
        default=os.getenv("API_EXTERNO_AGENT", 
        "https://gaiotto-flowiseai.hf.space/api/v1/prediction/126dd353-3c69-4304-9542-1263d07c711a")
    )
    
    # Server Configuration
    port: int = Field(default=int(os.getenv("PORT", "8000")))
    environment: str = Field(default=os.getenv("ENVIRONMENT", "development"))
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True


# Instância global
settings = Settings()