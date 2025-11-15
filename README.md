# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 1. Limpar cache do pip e reinstalar
pip cache purge
python -m pip install --upgrade --force-reinstall pip

# 2. Atualizar pip
pip install --upgrade pip

# 3. Instalar dependências
pip install --upgrade -r requirements.txt

# 4. Instalar dependências Node.js
npm install -g firecrawl-mcp

# 5. Copiar e configurar .env
cp .env.example .env
# Editar .env com suas chaves de API

# 6. Executar aplicação (desenvolvimento)
python app.py

# 7. Ou usar Docker
docker-compose up --build

# 8. Acessar aplicação
# http://localhost:8000