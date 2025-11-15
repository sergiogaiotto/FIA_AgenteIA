# agents/classifica_imagem_agent.py - Agente de Classificação de Imagens

# Agente ClassificaImagem - Análise de imagens usando LlamaIndex com GPT-4 Vision
# Especializado em análise visual, marketing e design

import os
import asyncio
import json
import aiohttp
from typing import Dict, Any, List, Optional
from io import BytesIO
import base64

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import encode_image

# Pydantic para modelos de dados
from pydantic import BaseModel, Field

# Carregamento de variáveis de ambiente
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


# ===============================
# MODELOS PYDANTIC
# ===============================

class ImageAnalysisRequest(BaseModel):
    """Modelo para requisições de análise de imagem"""
    image_url: str = Field(..., description="URL da imagem a ser analisada")
    analysis_type: str = Field(default="complete", description="Tipo de análise (complete, objects, colors, marketing)")
    custom_prompt: Optional[str] = Field(default=None, description="Prompt personalizado adicional")


class ObjectDetection(BaseModel):
    """Modelo para objetos detectados"""
    name: str = Field(..., description="Nome do objeto")
    confidence: float = Field(..., description="Nível de confiança da detecção (0-1)")
    description: str = Field(..., description="Descrição detalhada do objeto")
    position: Optional[str] = Field(default=None, description="Posição aproximada na imagem")


class ColorPalette(BaseModel):
    """Modelo para paleta de cores"""
    dominant_colors: List[str] = Field(..., description="Cores dominantes (hex codes)")
    color_harmony: str = Field(..., description="Tipo de harmonia de cores")
    mood: str = Field(..., description="Humor/sentimento transmitido pelas cores")
    accessibility: str = Field(..., description="Avaliação de acessibilidade das cores")


class MarketingInsights(BaseModel):
    """Modelo para insights de marketing"""
    target_audience: str = Field(..., description="Público-alvo sugerido")
    brand_positioning: str = Field(..., description="Posicionamento de marca")
    emotional_appeal: str = Field(..., description="Apelo emocional")
    call_to_action: str = Field(..., description="Sugestão de call-to-action")
    marketing_channels: List[str] = Field(..., description="Canais de marketing recomendados")


class ImageAnalysisResponse(BaseModel):
    """Modelo para resposta completa da análise"""
    image_url: str = Field(..., description="URL da imagem analisada")
    general_description: str = Field(..., description="Descrição geral da imagem")
    objects_detected: List[ObjectDetection] = Field(..., description="Objetos detectados na imagem")
    color_palette: ColorPalette = Field(..., description="Análise da paleta de cores")
    marketing_insights: MarketingInsights = Field(..., description="Insights de marketing")
    key_message: str = Field(..., description="Mensagem principal que a imagem transmite")
    composition_analysis: str = Field(..., description="Análise da composição visual")
    improvement_suggestions: List[str] = Field(..., description="Sugestões de melhoria")
    confidence_score: float = Field(..., description="Score geral de confiança da análise")


# ===============================
# PROMPTS ESPECIALIZADOS
# ===============================

class ImageAnalysisPrompts:
    """Container para prompts especializados em análise de imagem"""
    
    BASE_ANALYSIS_PROMPT = """
    Você é um especialista em análise visual, design e marketing digital. Analise esta imagem em detalhes e forneça insights profissionais.
    
    Sua análise deve incluir:
    
    1. DESCRIÇÃO GERAL:
    - Descreva o que você vê na imagem de forma detalhada
    - Identifique o contexto, ambiente e situação
    
    2. OBJETOS DETECTADOS:
    - Liste todos os objetos principais visíveis
    - Para cada objeto, forneça: nome, nível de confiança (0-1), descrição detalhada, posição aproximada
    
    3. ANÁLISE DE CORES:
    - Identifique as cores dominantes (forneça códigos hex aproximados)
    - Determine o tipo de harmonia de cores (monocromática, complementar, análoga, etc.)
    - Avalie o humor/sentimento transmitido pelas cores
    - Analise a acessibilidade das cores
    
    4. INSIGHTS DE MARKETING:
    - Sugira o público-alvo mais adequado
    - Recomende posicionamento de marca
    - Identifique o apelo emocional
    - Proponha call-to-action adequado
    - Sugira canais de marketing apropriados
    
    5. MENSAGEM PRINCIPAL:
    - Identifique a mensagem principal que a imagem transmite
    
    6. ANÁLISE DE COMPOSIÇÃO:
    - Avalie elementos como regra dos terços, simetria, enquadramento
    - Analise lighting, contraste, profundidade
    
    7. SUGESTÕES DE MELHORIA:
    - Forneça pelo menos 3 sugestões concretas para melhorar a imagem
    
    8. SCORE DE CONFIANÇA:
    - Forneça um score geral (0-1) da sua confiança na análise
    
    IMPORTANTE: Responda APENAS com um JSON válido seguindo exatamente esta estrutura:
    """
    
    OBJECTS_FOCUS_PROMPT = """
    Foque especificamente na detecção e análise de objetos nesta imagem. 
    Identifique todos os elementos visuais relevantes e forneça informações detalhadas sobre cada um.
    """
    
    COLORS_FOCUS_PROMPT = """
    Concentre-se na análise de cores desta imagem.
    Forneça uma análise detalhada da paleta de cores, harmonia e impacto psicológico.
    """
    
    MARKETING_FOCUS_PROMPT = """
    Analise esta imagem do ponto de vista de marketing e comunicação visual.
    Foque em insights estratégicos, público-alvo e oportunidades de marketing.
    """


# ===============================
# AGENTE PRINCIPAL
# ===============================

class ClassificaImagemAgent:
    """Agente especializado em análise de imagens com foco em marketing e design"""
    
    def __init__(self):
        """Inicializa agente de classificação de imagem"""
        
        # Validação de chaves de API
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        # Configuração do LlamaIndex
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_key,
            temperature=0.1
        )
        
        # Configuração do modelo multimodal
        self.multimodal_llm = OpenAIMultiModal(
            model="gpt-4o-mini",
            api_key=self.openai_key,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Prompts organizados
        self.prompts = ImageAnalysisPrompts()
        
        # Histórico de análises
        self.analysis_history: List[Dict[str, Any]] = []
        
        print("✅ ClassificaImagem Agent inicializado com LlamaIndex")
    
    async def download_image(self, image_url: str) -> bytes:
        """
        Baixa imagem da URL fornecida
        
        Args:
            image_url: URL da imagem
            
        Returns:
            Bytes da imagem baixada
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"Erro ao baixar imagem: HTTP {response.status}")
                    
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        raise Exception(f"URL não contém imagem válida. Content-Type: {content_type}")
                    
                    return await response.read()
                    
        except Exception as e:
            raise Exception(f"Erro ao baixar imagem: {str(e)}")
    
    def create_json_schema(self) -> str:
        """Cria schema JSON para estruturar a resposta"""
        return '''
        {
            "image_url": "string",
            "general_description": "string",
            "objects_detected": [
                {
                    "name": "string",
                    "confidence": 0.95,
                    "description": "string",
                    "position": "string"
                }
            ],
            "color_palette": {
                "dominant_colors": ["#hex1", "#hex2", "#hex3"],
                "color_harmony": "string",
                "mood": "string",
                "accessibility": "string"
            },
            "marketing_insights": {
                "target_audience": "string",
                "brand_positioning": "string",
                "emotional_appeal": "string",
                "call_to_action": "string",
                "marketing_channels": ["channel1", "channel2"]
            },
            "key_message": "string",
            "composition_analysis": "string",
            "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "confidence_score": 0.85
        }
        '''
    
    async def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResponse:
        """
        Analisa imagem usando GPT-4 Vision via LlamaIndex
        
        Args:
            request: Requisição com URL da imagem e parâmetros
            
        Returns:
            Análise completa da imagem
        """
        try:
            print(f"🔍 Analisando imagem: {request.image_url}")
            
            # Baixa a imagem
            image_data = await self.download_image(request.image_url)
            
            # Codifica imagem em base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Constrói prompt baseado no tipo de análise
            prompt = self.prompts.BASE_ANALYSIS_PROMPT
            
            if request.analysis_type == "objects":
                prompt += "\n" + self.prompts.OBJECTS_FOCUS_PROMPT
            elif request.analysis_type == "colors":
                prompt += "\n" + self.prompts.COLORS_FOCUS_PROMPT
            elif request.analysis_type == "marketing":
                prompt += "\n" + self.prompts.MARKETING_FOCUS_PROMPT
            
            # Adiciona prompt personalizado se fornecido
            if request.custom_prompt:
                prompt += f"\n\nPROMPT ADICIONAL: {request.custom_prompt}"
            
            # Adiciona schema JSON
            prompt += f"\n\nESTRUTURA JSON ESPERADA:\n{self.create_json_schema()}"
            
            # Prepara imagem para LlamaIndex
            from llama_index.core.schema import ImageDocument
            image_doc = ImageDocument(image=f"data:image/jpeg;base64,{image_base64}")
            
            # Faz análise usando multimodal LLM
            response = self.multimodal_llm.complete(
                prompt=prompt,
                image_documents=[image_doc]
            )
            
            # Processa resposta
            response_text = response.text.strip()
            
            # Tenta extrair JSON da resposta
            json_response = self._extract_json_from_response(response_text)
            
            # Valida e cria resposta estruturada
            analysis_response = self._create_structured_response(json_response, request.image_url)
            
            # Adiciona ao histórico
            self.analysis_history.append({
                "timestamp": asyncio.get_event_loop().time(),
                "request": request.model_dump(),
                "response": analysis_response.model_dump()
            })
            
            print(f"✅ Análise de imagem concluída com sucesso")
            return analysis_response
            
        except Exception as e:
            print(f"❌ Erro na análise da imagem: {e}")
            # Retorna resposta de erro estruturada
            return self._create_error_response(request.image_url, str(e))
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extrai JSON da resposta do LLM"""
        try:
            # Procura por JSON entre ```json e ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Procura por JSON entre { e }
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            # Se falhar, tenta criar estrutura básica a partir do texto
            return self._parse_text_to_json(response)
    
    def _parse_text_to_json(self, text: str) -> Dict[str, Any]:
        """Converte texto em estrutura JSON básica em caso de falha"""
        return {
            "image_url": "",
            "general_description": text[:500] + "..." if len(text) > 500 else text,
            "objects_detected": [
                {
                    "name": "Análise não estruturada",
                    "confidence": 0.5,
                    "description": "Resposta em formato de texto",
                    "position": "N/A"
                }
            ],
            "color_palette": {
                "dominant_colors": ["#000000", "#FFFFFF"],
                "color_harmony": "Não determinado",
                "mood": "Neutro",
                "accessibility": "Não avaliado"
            },
            "marketing_insights": {
                "target_audience": "Não determinado",
                "brand_positioning": "Não determinado",
                "emotional_appeal": "Neutro",
                "call_to_action": "Visualizar conteúdo",
                "marketing_channels": ["Digital", "Social Media"]
            },
            "key_message": "Análise necessita de refinamento",
            "composition_analysis": "Análise visual básica realizada",
            "improvement_suggestions": [
                "Reformular prompt para melhor estruturação",
                "Tentar novamente com imagem diferente",
                "Verificar qualidade da imagem"
            ],
            "confidence_score": 0.3
        }
    
    def _create_structured_response(self, json_data: Dict[str, Any], image_url: str) -> ImageAnalysisResponse:
        """Cria resposta estruturada a partir do JSON"""
        try:
            # Garante que image_url está preenchido
            json_data["image_url"] = image_url
            
            # Validação e preenchimento de campos obrigatórios
            required_fields = {
                "general_description": "Descrição não fornecida",
                "key_message": "Mensagem não identificada",
                "composition_analysis": "Análise de composição não realizada",
                "confidence_score": 0.5
            }
            
            for field, default in required_fields.items():
                if field not in json_data:
                    json_data[field] = default
            
            # Garante estruturas complexas
            if "objects_detected" not in json_data:
                json_data["objects_detected"] = []
            
            if "color_palette" not in json_data:
                json_data["color_palette"] = {
                    "dominant_colors": ["#000000"],
                    "color_harmony": "Não determinado",
                    "mood": "Neutro", 
                    "accessibility": "Não avaliado"
                }
            
            if "marketing_insights" not in json_data:
                json_data["marketing_insights"] = {
                    "target_audience": "Público geral",
                    "brand_positioning": "Neutro",
                    "emotional_appeal": "Informativo",
                    "call_to_action": "Saiba mais",
                    "marketing_channels": ["Digital"]
                }
            
            if "improvement_suggestions" not in json_data:
                json_data["improvement_suggestions"] = ["Análise mais detalhada necessária"]
            
            return ImageAnalysisResponse(**json_data)
            
        except Exception as e:
            print(f"Erro ao criar resposta estruturada: {e}")
            return self._create_error_response(image_url, str(e))
    
    def _create_error_response(self, image_url: str, error_message: str) -> ImageAnalysisResponse:
        """Cria resposta de erro estruturada"""
        return ImageAnalysisResponse(
            image_url=image_url,
            general_description=f"Erro na análise: {error_message}",
            objects_detected=[],
            color_palette=ColorPalette(
                dominant_colors=["#FF0000"],
                color_harmony="Erro",
                mood="Erro de processamento",
                accessibility="Não avaliável"
            ),
            marketing_insights=MarketingInsights(
                target_audience="Não determinado",
                brand_positioning="Erro de análise",
                emotional_appeal="Erro",
                call_to_action="Tentar novamente",
                marketing_channels=["Nenhum"]
            ),
            key_message="Erro na análise da imagem",
            composition_analysis="Análise não pôde ser realizada devido a erro",
            improvement_suggestions=[
                "Verificar URL da imagem",
                "Tentar com imagem diferente",
                "Verificar conectividade"
            ],
            confidence_score=0.0
        )
    
    async def process_message(self, user_message: str) -> str:
        """
        Processa mensagem do usuário para análise de imagem
        
        Args:
            user_message: Mensagem contendo URL da imagem
            
        Returns:
            Resultado da análise formatado
        """
        try:
            # Extrai URL da mensagem
            import re
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, user_message)
            
            if not urls:
                return "❌ Por favor, forneça uma URL válida de imagem para análise.\n\nExemplo: 'Analise esta imagem: https://exemplo.com/imagem.jpg'"
            
            image_url = urls[0]
            
            # Determina tipo de análise baseado na mensagem
            analysis_type = "complete"
            if "objetos" in user_message.lower() or "objeto" in user_message.lower():
                analysis_type = "objects"
            elif "cores" in user_message.lower() or "cor" in user_message.lower():
                analysis_type = "colors"
            elif "marketing" in user_message.lower():
                analysis_type = "marketing"
            
            # Cria requisição
            request = ImageAnalysisRequest(
                image_url=image_url,
                analysis_type=analysis_type,
                custom_prompt=user_message
            )
            
            # Realiza análise
            response = await self.analyze_image(request)
            
            # Formata resposta
            return self._format_analysis_response(response)
            
        except Exception as e:
            return f"❌ Erro ao processar análise: {str(e)}"
    
    def _format_analysis_response(self, response: ImageAnalysisResponse) -> str:
        """Formata resposta da análise para exibição"""
        formatted = f"""🖼️ **Análise de Imagem Completa**

📸 **URL:** {response.image_url}

📝 **Descrição Geral:**
{response.general_description}

🎯 **Mensagem Principal:**
{response.key_message}

🔍 **Objetos Detectados ({len(response.objects_detected)}):**"""
        
        for i, obj in enumerate(response.objects_detected, 1):
            formatted += f"\n{i}. **{obj.name}** (Confiança: {obj.confidence:.0%})"
            formatted += f"\n   - {obj.description}"
            if obj.position:
                formatted += f"\n   - Posição: {obj.position}"
        
        formatted += f"""

🎨 **Paleta de Cores:**
- **Cores Dominantes:** {', '.join(response.color_palette.dominant_colors)}
- **Harmonia:** {response.color_palette.color_harmony}
- **Humor:** {response.color_palette.mood}
- **Acessibilidade:** {response.color_palette.accessibility}

📈 **Insights de Marketing:**
- **Público-Alvo:** {response.marketing_insights.target_audience}
- **Posicionamento:** {response.marketing_insights.brand_positioning}
- **Apelo Emocional:** {response.marketing_insights.emotional_appeal}
- **Call-to-Action:** {response.marketing_insights.call_to_action}
- **Canais Recomendados:** {', '.join(response.marketing_insights.marketing_channels)}

🎨 **Análise de Composição:**
{response.composition_analysis}

💡 **Sugestões de Melhoria:**"""
        
        for i, suggestion in enumerate(response.improvement_suggestions, 1):
            formatted += f"\n{i}. {suggestion}"
        
        formatted += f"""

📊 **Score de Confiança:** {response.confidence_score:.0%}

```json
{json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}
```"""
        
        return formatted
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o agente"""
        return {
            "name": "ClassificaImagem Agent",
            "version": "1.0.0",
            "type": "image_analysis",
            "capabilities": [
                "Análise visual de imagens",
                "Detecção de objetos",
                "Análise de paleta de cores",
                "Insights de marketing",
                "Análise de composição",
                "Sugestões de melhoria"
            ],
            "supported_formats": ["JPG", "PNG", "WEBP", "GIF"],
            "analysis_types": ["complete", "objects", "colors", "marketing"],
            "framework": "LlamaIndex + GPT-4 Vision",
            "analyses_performed": len(self.analysis_history)
        }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de análises"""
        return self.analysis_history.copy()
    
    def reset_conversation(self):
        """Reseta histórico de conversas"""
        self.analysis_history = []
        print("🔄 Histórico de análises resetado")