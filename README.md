# 🤖 Adaptando o Exemplo do Oracle Agent AI para Utilizar o MCP

Este tutorial mostra como adaptar o exemplo do [tutorial oficial de Agent AI](https://docs.oracle.com/en/learn/oci-agent-ai/) para o padrão **MCP (Multi-Agent Communication Protocol)**. O MCP permite uma separação mais clara entre ferramentas (ferramentas de negócios) e o modelo de linguagem, seguindo um protocolo universal de comunicação entre agentes.

## ✅ O que você vai aprender

- Como refatorar o backend para rodar como um **servidor MCP**
- Como utilizar o LangChain + LangGraph para se conectar com este servidor
- Como estruturar ferramentas de negócios compatíveis com MCP
- Como criar um agente de pedidos inteligente que entende comandos como:
  - "Quero dois refrigerantes"
  - "Qual o total?"
  - "Entregar no CEP 01310-000 número 1000"

---

## 📁 Estrutura do Projeto

```
project/
│
├── server_mcp.py        # MCP Server com ferramentas de negócios
└── client_mcp.py        # Cliente LangGraph que usa as ferramentas via MCP
```

---

## 🧠 Passo 1: Criando o Servidor MCP (`server_mcp.py`)

### 1. Instale o pacote `mcp` e suas dependências:

```bash
pip install mcp fastapi langchain
```

### 2. Crie o arquivo `server_mcp.py` com o seguinte conteúdo:

> 💡 Esse servidor registra ferramentas como `insert_order`, `order_cost`, `delivery_address`, etc.

```python
# server_mcp.py

import sqlite3
import os
import requests
from requests.auth import HTTPBasicAuth
from mcp.server.fastmcp import FastMCP

user_name = "SEU_EMAIL"
password = "SUA_SENHA"

mcp = FastMCP("Restaurant")

# ---------- Banco de Dados Local ----------
def connect_db():
    return sqlite3.connect('orders.db')

def create_orders_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_item_to_db(item):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (item) VALUES (?)", (item,))
    conn.commit()
    conn.close()

def delete_item_from_db(item):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM orders WHERE item = ?", (item,))
    conn.commit()
    conn.close()

def get_all_items_from_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT item FROM orders")
    items = cursor.fetchall()
    conn.close()
    return [item[0] for item in items]

# ---------- Ferramentas MCP ----------
@mcp.tool()
def insert_order(item):
    """Adiciona um item no pedido atual."""
    insert_item_to_db(item)
    return {"message": f"Item '{item}' adicionado com sucesso!"}

@mcp.tool()
def delete_order(item):
    """Remove um item do pedido atual."""
    delete_item_from_db(item)
    return {"message": f"Item '{item}' removido com sucesso!"}

@mcp.tool()
def search_order():
    """Mostra os itens atuais no pedido."""
    return {"items": get_all_items_from_db()}

@mcp.tool()
def order_cost():
    """Calcula o custo total do pedido."""
    items = get_all_items_from_db()
    total = len(items) * 10
    return {"total_cost": total}

@mcp.tool()
def delivery_address(postalCode: str, number: str = "", complement: str = "") -> str:
    """Busca o endereço completo baseado no CEP."""
    url = f"https://SEU_API_GATEWAY/cep/cep?cep={postalCode}"
    response = requests.get(url, auth=HTTPBasicAuth(user_name, password))
    if not response.ok:
        return "Erro ao buscar endereço"

    json_data = response.json()
    address = json_data.get("frase", "Endereço não encontrado")
    return f"{address}, Número: {number}, Complemento: {complement}"

# ---------- Execução ----------
if __name__ == "__main__":
    os.remove("orders.db") if os.path.exists("orders.db") else None
    create_orders_table()
    mcp.run(transport="stdio")
```

---

## 💬 Passo 2: Criando o Cliente com LangGraph (`client_mcp.py`)

### 1. Instale as dependências:

```bash
pip install langgraph langchain langchain-mcp-adapters
```

### 2. Crie o arquivo `client_mcp.py` com o seguinte conteúdo:

```python
# client_mcp.py

import asyncio
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

class MemoryState:
    def __init__(self):
        self.messages = []

llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="SEU_OCID_COMPARTMENT",
    auth_profile="SEU_PROFILE",
    model_kwargs={"temperature": 0.1, "top_p": 0.75, "max_tokens": 2000}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente de pedidos.
    Use os comandos MCP conforme necessário: insert_order, delete_order, search_order, order_cost, delivery_address.
    O cliente pode pedir coisas como 'quero dois refrigerantes', ou 'qual o total?', ou 'entregar no CEP 01310-000 número 1000'."""),
    ("placeholder", "{messages}")
])

server_params = StdioServerParameters(
    command="python",
    args=["server_mcp.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            if not tools:
                print("❌ Ferramentas MCP não carregadas. Verifique se o servidor está rodando.")
                return

            memory = MemoryState()
            agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

            print("🤖 MCP Agent pronto!")
            while True:
                user_input = input("Você: ")
                if user_input.lower() in ["sair", "exit", "quit"]:
                    break
                if not user_input.strip():
                    continue

                memory.messages.append(HumanMessage(content=user_input))
                result = await agent.ainvoke({"messages": memory.messages})
                new_msgs = result.get("messages", [])
                memory.messages.extend(new_msgs)
                print("Assistente:", new_msgs[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔪 Executando o Projeto

1. **Terminal 1** – Rode o cliente:
```bash
python client_mcp.py
```

2. O servidor será iniciado automaticamente via `StdioServerParameters`, e o cliente começará a conversar com o agente.

---

## 🔧 Exemplos de Comandos

- **"Quero uma pizza e um refrigerante"** → `insert_order`
- **"Qual o total?"** → `order_cost`
- **"Remova o refrigerante"** → `delete_order`
- **"O que eu pedi?"** → `search_order`
- **"Entregar no CEP 01310-000 número 1000"** → `delivery_address`

---

## 🧰 Considerações Finais

O MCP permite um modelo plugável e desacoplado onde diferentes servidores MCP com ferramentas especializadas podem ser utilizados por qualquer agente. Isso favorece a **escalabilidade**, **padronização** e **reaproveitamento** de ferramentas.

Você pode agora estender este exemplo adicionando ferramentas como `pagamento`, `status do pedido`, ou até conectar com APIs reais de entrega 🚀

