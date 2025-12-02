# 📮 Postman Collection - TDvX API

## 🚀 Início Rápido

### 1. Importar no Postman

1. Abra o Postman
2. Clique em **Import** (canto superior esquerdo)
3. Selecione o arquivo `TDvX_API.postman_collection.json`
4. Clique em **Import**

✅ Pronto! A collection está importada com todos os endpoints.

### 2. Primeiro Uso

#### Passo 1: Criar sua primeira API Key

1. Abra a pasta **Authentication & Keys**
2. Clique em **Create API Key**
3. Clique em **Send**
4. 🎉 A API key será salva automaticamente nas variáveis!
5. ⚠️ **IMPORTANTE**: Copie a `api_key` da resposta - ela só aparece uma vez!

```json
{
  "api_key": "sk_tdvx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

#### Passo 2: Testar um endpoint protegido

1. Abra **Transcription** → **List Available Models**
2. Clique em **Send**
3. ✅ Deve retornar a lista de modelos disponíveis

#### Passo 3: Transcrever um áudio

1. Abra **Transcription** → **Transcribe Audio File**
2. Na aba **Body**, clique em **Select Files** no campo `file`
3. Escolha um arquivo de áudio (WAV, MP3, M4A ou FLAC)
4. Clique em **Send**
5. ✅ Receberá a transcrição com identificação de falantes

## 📁 Estrutura da Collection

```
TDvX Transcription API
├── 📁 Authentication & Keys (Requer master key)
│   ├── Create API Key ➜ Criar nova chave
│   ├── List API Keys ➜ Listar todas as chaves
│   ├── Get API Key Details ➜ Ver detalhes e estatísticas
│   ├── Update Rate Limit ➜ Atualizar limite de requisições
│   └── Revoke API Key ➜ Revogar uma chave
│
├── 📁 Transcription (Requer API key)
│   ├── List Available Models ➜ Ver modelos disponíveis
│   ├── Transcribe Audio File ➜ Transcrever (modo rápido)
│   ├── Transcribe File (High Quality) ➜ Máxima qualidade
│   └── Transcribe File (Balanced) ➜ Qualidade balanceada
│
├── 📁 Public Endpoints (Sem autenticação)
│   ├── Health Check ➜ Verificar se servidor está rodando
│   └── API Documentation ➜ Documentação Swagger
│
└── 📁 Testing & Examples (Testes automatizados)
    ├── Test - No Auth ➜ Deve falhar sem autenticação
    ├── Test - Invalid Key ➜ Deve falhar com chave inválida
    └── Test - Rate Limit ➜ Testar limite de requisições
```

## 🔑 Variáveis

A collection vem com variáveis pré-configuradas:

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `base_url` | `http://localhost:8000` | URL do servidor |
| `master_key` | `master_tdvx_...` | Chave mestra (gerenciar API keys) |
| `api_key` | *(vazio)* | Sua API key (preenchida automaticamente) |
| `key_id` | *(vazio)* | ID da sua key (preenchido automaticamente) |

### Como editar variáveis:

1. Clique nos **...** da collection
2. Selecione **Edit**
3. Vá para aba **Variables**
4. Edite os valores

## ✨ Recursos da Collection

### 🔄 Salvamento Automático

Quando você cria uma API key:
- ✅ `api_key` é salva automaticamente
- ✅ `key_id` é salvo automaticamente
- ✅ Valores aparecem no Console

### 🧪 Testes Automáticos

Cada requisição inclui testes que:
- ✅ Validam códigos de resposta
- ✅ Verificam campos obrigatórios
- ✅ Mostram informações no Console

Veja os resultados na aba **Test Results**.

### 📊 Rastreamento de Rate Limit

Execute **Test - Rate Limit** para ver:
- Quantas requisições restam
- Quando o limite reseta
- Headers de rate limit

## 🎯 Exemplos de Uso

### Criar uma chave de produção

```
POST http://localhost:8000/api/keys
Authorization: Bearer {{master_key}}

{
  "name": "Chave de Produção",
  "description": "Para uso em produção",
  "rate_limit_per_hour": 1000
}
```

### Listar todas as chaves (incluindo revogadas)

```
GET http://localhost:8000/api/keys?include_revoked=true
Authorization: Bearer {{master_key}}
```

### Atualizar limite de requisições

```
PATCH http://localhost:8000/api/keys/1/rate-limit
Authorization: Bearer {{master_key}}

{
  "rate_limit_per_hour": 200
}
```

### Transcrever com modelo específico

```
POST http://localhost:8000/transcribe?model=tdv1
Authorization: Bearer {{api_key}}
[arquivo de áudio]
```

## 🧪 Cenários de Teste

### Teste 1: Sem Autenticação ❌

Execute **Test - No Auth (Should Fail)**
- Deve retornar `401 Unauthorized`
- Prova que a autenticação está funcionando

### Teste 2: Chave Inválida ❌

Execute **Test - Invalid Key (Should Fail)**
- Deve retornar `401 Unauthorized`
- Prova que a validação está funcionando

### Teste 3: Rate Limiting 🔢

Execute **Test - Rate Limit** 100+ vezes
- Primeiras 100: `200 OK`
- Depois: `429 Too Many Requests`
- Veja headers `X-RateLimit-*`

### Teste 4: Workflow Completo ✅

1. Criar chave
2. Listar chaves
3. Ver detalhes
4. Atualizar limite
5. Usar a chave
6. Revogar chave
7. Testar chave revogada (deve falhar)

## 🛠️ Dicas

### 💡 Executar múltiplos testes

1. Clique nos **...** da collection
2. Selecione **Run collection**
3. Escolha os testes
4. Clique em **Run**

### 💡 Copiar como cURL

1. Clique em **Code** (ao lado de Send)
2. Selecione **cURL**
3. Copie e use no terminal

### 💡 Verificar rate limits

1. Vá em **Response** → **Headers**
2. Procure por `X-RateLimit-*`
3. Monitore seu uso

## ⚠️ Solução de Problemas

### "Could not send request"
**Problema**: Servidor não está rodando

**Solução**:
```bash
cd C:\Users\Usuario\war\tdvx
venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### "401 Unauthorized"
**Problema**: API key não configurada ou inválida

**Solução**:
1. Verifique se `{{api_key}}` está preenchida
2. Crie uma nova chave se necessário
3. Verifique se a chave não foi revogada

### "429 Too Many Requests"
**Problema**: Limite de requisições excedido

**Solução**:
1. Aguarde o reset (próxima hora cheia)
2. Ou aumente o limite via **Update Rate Limit**
3. Ou crie nova chave com limite maior

### Upload de arquivo falha
**Problema**: Arquivo muito grande ou formato incorreto

**Solução**:
1. Verifique tamanho < 100MB
2. Formatos suportados: WAV, MP3, M4A, FLAC
3. Verifique logs do servidor

## 📚 Documentação

- **Detalhes completos**: `POSTMAN_GUIDE.md` (em inglês)
- **Autenticação**: `AUTHENTICATION.md`
- **Documentação API**: http://localhost:8000/docs (quando servidor estiver rodando)

## 🎉 Pronto!

Agora você pode:
- ✅ Criar e gerenciar API keys
- ✅ Transcrever áudios
- ✅ Testar rate limiting
- ✅ Monitorar uso

**Boa sorte!** 🚀

---

**Criado com**: Claude Code
**Data**: 2 de Dezembro de 2025
