# agents/rag_agent.py - Comentado Linha a Linha

# Agente RAG - Retrieval-Augmented Generation com Pinecone
# Comentário descritivo: agente especializado em pesquisa semântica com RAG

import os
# Módulo padrão para interação com sistema operacional

import json
# Módulo para manipulação de dados JSON

from typing import List, Dict, Any, Optional
# Type hints para melhor documentação e type safety

import asyncio
# Biblioteca para programação assíncrona

# Imports para embeddings e LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ChatOpenAI: modelo de chat da OpenAI
# OpenAIEmbeddings: geração de embeddings usando OpenAI

from langchain_core.messages import HumanMessage, SystemMessage
# Tipos de mensagem padronizados do LangChain

# Imports para RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Text splitter para dividir documentos em chunks
# Recursive: tenta manter contexto semântico

from langchain_community.vectorstores import Pinecone as LangChainPinecone
# Integração LangChain com Pinecone
# Abstração de alto nível para vector database

# Pinecone SDK
from pinecone import Pinecone, ServerlessSpec
# SDK oficial do Pinecone
# ServerlessSpec: configuração para mode serverless

# Pydantic para modelos de dados
from pydantic import BaseModel, Field
# BaseModel: validação e serialização
# Field: metadados para campos

# Carregamento de variáveis
from dotenv import load_dotenv
# Carregamento de configuração de arquivo .env

# Firecrawl para coleta de dados
from firecrawl import FirecrawlApp
# SDK para web scraping estruturado

# Carrega variáveis de ambiente
load_dotenv()


# ===============================
# MODELOS PYDANTIC
# ===============================

class RAGQuery(BaseModel):
    """Modelo para queries RAG"""
    query: str = Field(..., description="Consulta do usuário")
    # Campo obrigatório: pergunta do usuário
    # Field(...): campo obrigatório com metadados
    
    top_k: int = Field(default=4, description="Número de documentos similares")
    # Quantidade de documentos para retrieval
    # Default 5: balanceia relevância e velocidade
    
    threshold: float = Field(default=0.1, description="Threshold de similaridade")
    # Limite mínimo de similaridade semântica
    # 0.1: valor pouco conservador que filtra resultados irrelevantes (para podermos testar)


class RAGDocument(BaseModel):
    """Modelo para documentos indexados"""
    id: str = Field(..., description="ID único do documento")
    # Identificador único no Pinecone
    
    content: str = Field(..., description="Conteúdo do documento")
    # Texto completo do documento
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados")
    # Informações adicionais: URL, título, data, etc.
    # default_factory=dict: cria dict vazio por padrão
    
    score: Optional[float] = Field(default=None, description="Score de similaridade")
    # Score de similaridade semântica (0-1)
    # None por padrão, preenchido durante busca


class RAGResponse(BaseModel):
    """Modelo para resposta RAG"""
    answer: str = Field(..., description="Resposta gerada")
    # Resposta final do LLM com contexto
    
    sources: List[RAGDocument] = Field(default_factory=list, description="Documentos fonte")
    # Documentos usados para gerar resposta
    # Transparência: usuário vê fontes
    
    query: str = Field(..., description="Query original")
    # Query que gerou a resposta
    
    confidence: Optional[float] = Field(default=None, description="Confiança da resposta")
    # Nível de confiança baseado nos scores


# ===============================
# SERVIÇO PINECONE
# ===============================

class PineconeService:
    """Serviço para integração com Pinecone"""
    # Service class: encapsula operações do Pinecone
    # Abstração: isola complexidade da API Pinecone
    
    def __init__(self, index_name: str = "fia-agente-ia"):
        """Inicializa serviço Pinecone"""
        # index_name: nome do índice no Pinecone
        # Default: nome descritivo para base de conhecimento
        
        # Validação de API key
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY não encontrada nas variáveis de ambiente")
        # Fail-fast: falha imediatamente se não configurado
        
        # Inicializa cliente Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        # Cliente principal para operações Pinecone
        
        self.index_name = index_name
        # Armazena nome do índice
        
        # Configuração de embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            # Modelo otimizado: balanceia qualidade e custo
            # text-embedding-3-small: versão eficiente da OpenAI
            openai_api_key=os.getenv("OPENAI_API_KEY")
            # Chave da OpenAI para embeddings
        )
        
        # Dimensão dos embeddings
        self.dimension = 1536
        # text-embedding-3-small produz vetores de 1536 dimensões
        # Fixo: dimensão não pode mudar após criação do índice
        
        # Inicializa índice
        self.index = None
        # Será inicializado em setup_index()
        
    async def setup_index(self):
        """Configura índice Pinecone (cria se não existir)"""
        # Método async para setup inicial
        # Verifica/cria índice conforme necessário
        
        try:
            # Lista índices existentes
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            # list_indexes(): retorna todos os índices da conta
            # Extrai apenas nomes para verificação
            
            if self.index_name not in existing_indexes:
                # Cria índice se não existir
                print(f"Criando índice Pinecone: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    # Nome do índice
                    dimension=self.dimension,
                    # Dimensão dos vetores (deve ser consistente)
                    metric="cosine",
                    # Métrica de similaridade: cosine similarity
                    # Boa para embeddings de texto
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                    # Configuração serverless: custo baseado em uso
                    # AWS us-east-1: região com melhor performance
                )
                
                # Aguarda criação do índice
                print("Aguardando criação do índice...")
                await asyncio.sleep(10)
                # Sleep: índice precisa de tempo para ficar ready
            
            # Conecta ao índice
            self.index = self.pc.Index(self.index_name)
            # Objeto Index para operações CRUD
            print(f"✅ Conectado ao índice: {self.index_name}")
            
        except Exception as e:
            print(f"❌ Erro ao configurar Pinecone: {e}")
            raise
            # Re-raise: erro crítico que impede funcionamento
    
    async def add_documents(self, documents: List[RAGDocument]) -> bool:
        """Adiciona documentos ao índice"""
        # Batch insert de documentos
        # Retorna bool: sucesso/falha da operação
        
        if not self.index:
            await self.setup_index()
        # Lazy initialization: configura índice se necessário
        
        try:
            # Prepara vetores para inserção
            vectors = []
            # Lista de tuplas (id, embedding, metadata)
            
            for doc in documents:
                # Gera embedding do conteúdo
                embedding = await self._generate_embedding(doc.content)
                # Embedding: representação vetorial do texto
                
                # Prepara vetor para Pinecone
                vector = {
                    "id": doc.id,
                    # ID único: chave primária no Pinecone
                    "values": embedding,
                    # values: vetor de embeddings
                    "metadata": {
                        **doc.metadata,
                        # Spread existing metadata
                        "content": doc.content[:512],
                        # Texto truncado para metadados
                        # Pinecone tem limite de tamanho para metadata
                    }
                }
                vectors.append(vector)
            
            # Insere em batch
            self.index.upsert(vectors=vectors)
            # upsert: insert ou update se ID já existir
            # Batch operation: mais eficiente que inserções individuais
            
            print(f"✅ {len(documents)} documentos adicionados ao Pinecone")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao adicionar documentos: {e}")
            return False
            # Graceful degradation: retorna False em vez de falhar
    
    async def search(self, query: str, top_k: int = 4, threshold: float = 0.1) -> List[RAGDocument]:
        """Busca documentos similares à query"""
        # Semantic search: busca por similaridade semântica
        # top_k: quantidade de resultados
        # threshold: filtro de qualidade
        
        if not self.index:
            await self.setup_index()
        # Garante que índice está configurado
        
        try:
            # Gera embedding da query
            query_embedding = await self._generate_embedding(query)
            # Mesmo modelo usado para indexação
            # Consistência: embeddings comparáveis
            
            # Busca no Pinecone
            results = self.index.query(
                vector=query_embedding,
                # Vetor da query para comparação
                top_k=top_k,
                # Quantidade máxima de resultados
                include_metadata=True,
                # Inclui metadados na resposta
                include_values=False
                # Não inclui embeddings (economiza banda)
            )
            
            # Processa resultados
            documents = []
            for match in results.matches:
                # Filtra por threshold
                if match.score >= threshold:
                    # score: similaridade cosine (0-1)
                    
                    doc = RAGDocument(
                        id=match.id,
                        content=match.metadata.get("content", ""),
                        # Recupera conteúdo dos metadados
                        metadata=match.metadata,
                        score=match.score,
                        # Inclui score para ranking
                        threshold=threshold
                        # Inclui threshold para lembrar do valor definido
                    )
                    documents.append(doc)
            
            print(f"🔍 Encontrados {len(documents)} documentos relevantes")
            return documents
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
            # Lista vazia em caso de erro
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para texto"""
        # Método privado: uso interno
        # Async: operação pode ser lenta
        
        try:
            # Usa OpenAI Embeddings
            embedding = self.embeddings.embed_query(text)
            # embed_query: otimizado para queries (vs documentos)
            return embedding
            
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            # Fallback: vetor zero
            return [0.0] * self.dimension
            # Lista de zeros com dimensão correta
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do índice"""
        # Método de diagnóstico
        # Útil para monitoramento e debugging
        
        if not self.index:
            return {"status": "not_initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            # describe_index_stats(): metadados do índice
            
            return {
                "status": "active",
                "total_vectors": stats.total_vector_count,
                # Quantidade total de vetores indexados
                "dimension": stats.dimension,
                # Dimensão dos vetores
                "index_fullness": stats.index_fullness,
                # Percentual de utilização do índice
                "namespaces": len(stats.namespaces) if stats.namespaces else 0
                # Quantidade de namespaces (organização lógica)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# ===============================
# AGENTE RAG PRINCIPAL
# ===============================

class RAGAgent:
    """Agente especializado em Retrieval-Augmented Generation"""
    # Classe principal que orquestra RAG workflow
    # Combina: retrieval (Pinecone) + generation (OpenAI)
    
    def __init__(self, index_name: str = "fia-agente-ia"):
        """Inicializa agente RAG"""
        
        # Validação de chaves de API
        required_keys = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
        for key in required_keys:
            if not os.getenv(key):
                raise ValueError(f"{key} não encontrada nas variáveis de ambiente")
        # Validação em loop: todas as chaves necessárias
        
        # Inicializa serviços
        self.pinecone_service = PineconeService(index_name)
        # Serviço de vector database
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            # Modelo otimizado: boa qualidade, custo menor
            temperature=0.0,
            # Baixa temperatura: respostas mais determinísticas
            # RAG precisa de consistência e precisão
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configuração de text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            # Tamanho do chunk: balance entre contexto e performance
            # 512 chars: suficiente para parágrafos completos
            chunk_overlap=128,
            # Overlap: mantém continuidade semântica
            # 128 chars: overlap significativo mas não excessivo
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            # Ordem de prioridade para quebras
            # Prioriza quebras semânticas (parágrafos, frases)
        )
        
        # Firecrawl para coleta de dados
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=firecrawl_key) if firecrawl_key else None
        # Opcional: permite funcionar sem Firecrawl
        
        # Sistema de prompt
        self.system_prompt = """Você é um assistente especializado em RAG (Retrieval-Augmented Generation).

                            Sua função é:
                            1. Analisar a pergunta do usuário
                            2. Usar os documentos fornecidos como contexto
                            3. Gerar respostas precisas e bem fundamentadas
                            4. Citar as fontes utilizadas
                            
                            Diretrizes:
                            - Base suas respostas APENAS nos documentos fornecidos
                            - Se não tiver informação suficiente, seja honesto sobre isso
                            - Cite específicamente as fontes relevantes
                            - Mantenha respostas claras e objetivas
                            - Use formatação markdown quando apropriado
                            
                            Sempre inclua uma seção "Fontes:" no final da resposta."""
                            # System prompt específico para RAG
                            # Enfatiza: precisão, citação de fontes, honestidade
    
    async def initialize(self):
        """Inicializa agente (configura Pinecone)"""
        # Método de inicialização async
        # Separado do __init__ para operações async
        
        await self.pinecone_service.setup_index()
        print("✅ RAG Agent inicializado")
    
    async def add_knowledge_from_url(self, url: str) -> bool:
        """Adiciona conhecimento a partir de URL"""
        # Método para expandir base de conhecimento
        # URL → scraping → chunking → embedding → indexing
        
        if not self.firecrawl:
            print("❌ Firecrawl não configurado")
            return False
        
        try:
            print(f"🌐 Fazendo scraping de: {url}")
            
            # Scraping da URL
            scraped = self.firecrawl.scrape_url(url, formats=["markdown"])
            # Markdown: formato estruturado, fácil de processar
            
            if not scraped or not hasattr(scraped, 'markdown'):
                print("❌ Falha no scraping")
                return False
            
            content = scraped.markdown
            # Conteúdo extraído em markdown
            
            # Chunking do conteúdo
            chunks = self.text_splitter.split_text(content)
            # Divide em pedaços processáveis
            
            # Cria documentos RAG
            documents = []
            for i, chunk in enumerate(chunks):
                doc = RAGDocument(
                    id=f"{url}#{i}",
                    # ID único: URL + índice do chunk
                    content=chunk,
                    metadata={
                        "source_url": url,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "scraped_at": str(asyncio.get_event_loop().time())
                        # Timestamp para tracking
                    }
                )
                documents.append(doc)
            
            # Adiciona ao Pinecone
            success = await self.pinecone_service.add_documents(documents)
            
            if success:
                print(f"✅ {len(documents)} chunks adicionados da URL: {url}")
            
            return success
            
        except Exception as e:
            print(f"❌ Erro ao adicionar conhecimento: {e}")
            return False
    
    async def add_knowledge_from_text(self, text: str, source_id: str) -> bool:
        """Adiciona conhecimento a partir de texto"""
        # Método alternativo: texto direto sem scraping
        # source_id: identificador da fonte
        
        try:
            # Chunking do texto
            chunks = self.text_splitter.split_text(text)
            
            # Cria documentos
            documents = []
            for i, chunk in enumerate(chunks):
                doc = RAGDocument(
                    id=f"{source_id}#{i}",
                    content=chunk,
                    metadata={
                        "source_id": source_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "added_at": str(asyncio.get_event_loop().time())
                    }
                )
                documents.append(doc)
            
            # Adiciona ao Pinecone
            success = await self.pinecone_service.add_documents(documents)
            
            if success:
                print(f"✅ {len(documents)} chunks adicionados do texto: {source_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Erro ao adicionar texto: {e}")
            return False
    
    async def query(self, user_query: str, top_k: int = 4, threshold: float = 0.1) -> RAGResponse:
        """Processa query usando RAG"""
        # Método principal: implementa pipeline RAG completo
        # Query → Retrieval → Augmentation → Generation
        
        try:
            print(f"🔍 Processando query: {user_query}")
            
            # 1. RETRIEVAL: Busca documentos relevantes
            relevant_docs = await self.pinecone_service.search(
                query=user_query,
                top_k=top_k,
                threshold=threshold
            )
            
            if not relevant_docs:
                # Fallback quando não há documentos relevantes
                return RAGResponse(
                    answer="Não encontrei informações relevantes na base de conhecimento para responder sua pergunta. Tente reformular ou adicionar mais contexto.",
                    sources=[],
                    query=user_query,
                    confidence=0.0
                )
            
            # 2. AUGMENTATION: Prepara contexto
            context = self._build_context(relevant_docs)
            # Combina documentos em contexto unificado
            
            # 3. GENERATION: Gera resposta com LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""Contexto dos documentos:
                {context}
                Pergunta do usuário: {user_query}
                Por favor, responda baseando-se apenas nas informações fornecidas no contexto.""")
            ]
            
            response = self.llm.invoke(messages)
            # LLM gera resposta com contexto aumentado
            
            # Calcula confiança baseada nos scores
            confidence = self._calculate_confidence(relevant_docs)
            
            return RAGResponse(
                answer=response.content,
                sources=relevant_docs,
                query=user_query,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"❌ Erro no RAG query: {e}")
            return RAGResponse(
                answer=f"Erro ao processar consulta: {str(e)}",
                sources=[],
                query=user_query,
                confidence=0.0
            )
    
    def _build_context(self, documents: List[RAGDocument]) -> str:
        """Constrói contexto a partir dos documentos"""
        # Método privado: combina documentos em texto unificado
        # Otimiza formato para consumo do LLM
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Enumera documentos para referência
            
            context_part = f"""--- Documento {i} (Score: {doc.score:.3f}) ---
            Fonte: {doc.metadata.get('source_url', doc.metadata.get('source_id', 'Desconhecido'))}
            Conteúdo: {doc.content}
            """
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
        # Junta com separadores claros
    
    def _calculate_confidence(self, documents: List[RAGDocument]) -> float:
        """Calcula confiança baseada nos scores dos documentos"""
        # Heurística: confiança baseada na qualidade dos matches
        
        if not documents:
            return 0.0
        
        # Média ponderada dos scores
        scores = [doc.score for doc in documents if doc.score]
        if not scores:
            return 0.5  # Default quando scores não disponíveis
        
        # Peso maior para o melhor resultado
        weighted_avg = sum(score * (len(scores) - i) for i, score in enumerate(scores)) / sum(range(1, len(scores) + 1))
        
        return min(weighted_avg, 1.0)
        # Garante que não excede 1.0
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da base de conhecimento"""
        # Método de diagnóstico
        
        pinecone_stats = await self.pinecone_service.get_stats()
        # Stats do Pinecone
        
        return {
            "rag_agent": {
                "status": "active",
                "model": "gpt-4.1-mini",
                "embedding_model": "text-embedding-3-small"
            },
            "knowledge_base": pinecone_stats,
            "capabilities": [
                "Semantic search",
                "Document chunking", 
                "Source citation",
                "Confidence scoring"
            ]
        }
    
    async def suggest_knowledge_sources(self, domain: str) -> List[str]:
        """Sugere fontes de conhecimento para um domínio"""
        # Feature: sugestão de URLs relevantes para indexar
        # domain: área de conhecimento (ex: "machine learning")
        
        if not self.firecrawl:
            return []
        
        try:
            # Busca fontes relevantes
            search_query = f"{domain} documentation tutorial guide"
            results = self.firecrawl.search(search_query, limit=4)
            
            if hasattr(results, 'data') and results.data:
                return [result.get('url', '') for result in results.data if result.get('url')]
            
            return []
            
        except Exception as e:
            print(f"❌ Erro ao sugerir fontes: {e}")
            return []