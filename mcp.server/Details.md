# 📘 MCP Server အကြောင်း အပြည့်အစုံ

## 📌 MCP ဆိုတာ ဘာလဲ?

**MCP (Model Context Protocol)** ဆိုတာ AI application များကို external system များနဲ့ ချိတ်ဆက်ဖို့ အတွက် ဖန်တီးထားတဲ့ **open-source standard protocol** တစ်ခုဖြစ်ပါတယ်။

MCP ကို **USB-C port** နဲ့ နှိုင်းယှဉ်လို့ရပါတယ်။ USB-C ဟာ electronic device အမျိုးမျိုးကို standardized ပုံစံတစ်မျိုးတည်းနဲ့ ချိတ်ဆက်ပေးသလိုမျိုး — MCP ဟာလည်း AI application များ (Claude, ChatGPT စတဲ့) ကို external system များ (databases, files, APIs, search engines) နဲ့ **standardized ပုံစံတစ်မျိုးတည်းနဲ့** ချိတ်ဆက်ပေးပါတယ်။

---

## 🏗️ Architecture Overview

### Participants (ပါဝင်သူများ)

MCP ဟာ **Client-Server Architecture** ကို အခြေခံထားပါတယ်။ အဓိက ပါဝင်သူ ၃ ဦးရှိပါတယ်:

| Participant | ရှင်းလင်းချက် |
|---|---|
| **MCP Host** | AI application (ဥပမာ - Claude Desktop, VS Code) — MCP Client တစ်ခု သို့မဟုတ် အများကြီးကို manage လုပ်ပေးတဲ့ application |
| **MCP Client** | MCP Server တစ်ခုနဲ့ dedicated connection ထားရှိပြီး context ကို ရယူပေးတဲ့ component |
| **MCP Server** | MCP Client များကို context (data, tools, prompts) ပေးတဲ့ program |

```
┌─────────────────────────────────────────────┐
│            MCP Host (AI Application)         │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ MCP      │  │ MCP      │  │ MCP      │   │
│  │ Client 1 │  │ Client 2 │  │ Client 3 │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │         │
└───────┼──────────────┼──────────────┼─────────┘
        │              │              │
   ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
   │ MCP      │  │ MCP      │  │ MCP      │
   │ Server A │  │ Server B │  │ Server C │
   │ (Local)  │  │ (Local)  │  │ (Remote) │
   │ e.g.     │  │ e.g.     │  │ e.g.     │
   │Filesystem│  │ Database │  │ Sentry   │
   └──────────┘  └──────────┘  └──────────┘
```

---

## 📚 Layers (အလွှာများ)

MCP ဟာ အလွှာ ၂ ခုနဲ့ ဖွဲ့စည်းထားပါတယ်:

### 1️⃣ Data Layer (အတွင်းအလွှာ)

JSON-RPC 2.0 ကို အခြေခံတဲ့ protocol ဖြစ်ပြီး အောက်ပါတို့ကို define လုပ်ပေးပါတယ်:

- **Lifecycle Management** — Connection initialization, capability negotiation, connection termination
- **Server Features** — Tools, Resources, Prompts
- **Client Features** — Sampling, Elicitation, Logging
- **Utility Features** — Notifications, Progress Tracking

### 2️⃣ Transport Layer (အပြင်အလွှာ)

Client နဲ့ Server ကြား communication channel ကို manage လုပ်ပေးပါတယ်။ Transport ၂ မျိုးရှိပါတယ်:

| Transport | ရှင်းလင်းချက် | သုံးရမယ့်နေရာ |
|---|---|---|
| **STDIO** | Standard Input/Output streams သုံးပြီး process communication လုပ်ပါတယ် | Local MCP Server |
| **Streamable HTTP** | HTTP POST + Server-Sent Events (SSE) သုံးပါတယ် | Remote MCP Server |

---

## ⚙️ Core Primitives (အဓိက အစိတ်အပိုင်းများ)

MCP Server ကနေ expose လုပ်နိုင်တဲ့ primitive ၃ မျိုးရှိပါတယ်:

### 🔧 1. Tools (ကိရိယာများ)

AI application ကနေ invoke (ခေါ်ယူ) လုပ်နိုင်တဲ့ **executable functions** တွေဖြစ်ပါတယ်။

**သုံးနိုင်တဲ့နေရာများ:**
- File operations (ဖိုင်ဖတ်ခြင်း/ရေးခြင်း)
- API calls (API များခေါ်ခြင်း)
- Database queries (Database queries လုပ်ခြင်း)
- Computations (တွက်ချက်မှုများ)

**Tool Definition:**
```json
{
  "name": "get_weather",
  "title": "Weather Information Provider",
  "description": "Get current weather information for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or zip code"
      }
    },
    "required": ["location"]
  }
}
```

**Protocol Methods:**
| Method | ရှင်းလင်းချက် |
|---|---|
| `tools/list` | ရနိုင်တဲ့ tool အားလုံးကို ရှာဖွေခြင်း |
| `tools/call` | Tool တစ်ခုကို execute (ခေါ်ယူ) ခြင်း |
| `notifications/tools/list_changed` | Tool list ပြောင်းလဲကြောင်း notification |

**Tool Call Request နမူနာ:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "Yangon"
    }
  }
}
```

**Tool Call Response နမူနာ:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in Yangon:\nTemperature: 95°F\nConditions: Partly cloudy"
      }
    ],
    "isError": false
  }
}
```

**Tool Result Content Types:**
| Content Type | ရှင်းလင်းချက် |
|---|---|
| `text` | Plain text response |
| `image` | Base64-encoded image data |
| `audio` | Base64-encoded audio data |
| `resource_link` | Resource URI link |
| `resource` | Embedded resource data |

---

### 📦 2. Resources (အရင်းအမြစ်များ)

AI application များကို **contextual information** ပေးနိုင်တဲ့ data sources တွေဖြစ်ပါတယ်။

**သုံးနိုင်တဲ့နေရာများ:**
- File contents (ဖိုင်အကြောင်းအရာ)
- Database records
- API responses
- Application-specific information

**Resource Definition:**
```json
{
  "uri": "file:///project/src/main.rs",
  "name": "main.rs",
  "title": "Rust Application Main File",
  "description": "Primary application entry point",
  "mimeType": "text/x-rust"
}
```

**Protocol Methods:**
| Method | ရှင်းလင်းချက် |
|---|---|
| `resources/list` | ရနိုင်တဲ့ resource အားလုံးကို list လုပ်ခြင်း |
| `resources/read` | Resource content ကို ဖတ်ခြင်း |
| `resources/templates/list` | Resource template များကို list လုပ်ခြင်း |
| `resources/subscribe` | Resource ပြောင်းလဲမှုကို subscribe လုပ်ခြင်း |
| `notifications/resources/list_changed` | Resource list ပြောင်းလဲကြောင်း notification |
| `notifications/resources/updated` | Resource update ကြောင်း notification |

**Resource Contents:**
- **Text Content** — `text` field ဖြင့် plain text data
- **Binary Content** — `blob` field ဖြင့် base64-encoded binary data

**Common URI Schemes:**
| URI Scheme | ရှင်းလင်းချက် |
|---|---|
| `https://` | Web ပေါ်မှ resource |
| `file://` | Filesystem-like resource |
| `git://` | Git version control integration |
| Custom | RFC3986 အတိုင်း custom URI scheme |

---

### 💬 3. Prompts (Prompt Template များ)

Language model များနဲ့ interaction ကို structure လုပ်ပေးနိုင်တဲ့ **reusable templates** တွေဖြစ်ပါတယ်။

**သုံးနိုင်တဲ့နေရာများ:**
- System prompts
- Few-shot examples
- Code review templates
- Specialized interaction patterns

**Prompt Definition:**
```json
{
  "name": "code_review",
  "title": "Request Code Review",
  "description": "Asks the LLM to analyze code quality and suggest improvements",
  "arguments": [
    {
      "name": "code",
      "description": "The code to review",
      "required": true
    }
  ]
}
```

**Protocol Methods:**
| Method | ရှင်းလင်းချက် |
|---|---|
| `prompts/list` | ရနိုင်တဲ့ prompt အားလုံးကို list လုပ်ခြင်း |
| `prompts/get` | Prompt content ကို ရယူခြင်း |
| `notifications/prompts/list_changed` | Prompt list ပြောင်းလဲကြောင်း notification |

**PromptMessage Content Types:**
- Text Content
- Image Content (base64-encoded)
- Audio Content (base64-encoded)
- Embedded Resources

---

## 🔄 Lifecycle Management

MCP ဟာ **stateful protocol** ဖြစ်တဲ့အတွက် lifecycle management လိုအပ်ပါတယ်။

### Connection Flow:

```
┌──────────┐                          ┌──────────┐
│  Client  │                          │  Server  │
└────┬─────┘                          └────┬─────┘
     │                                     │
     │  1. initialize (request)            │
     │  ──────────────────────────────────► │
     │  - protocolVersion                  │
     │  - capabilities                     │
     │  - clientInfo                       │
     │                                     │
     │  2. initialize (response)           │
     │  ◄────────────────────────────────── │
     │  - protocolVersion                  │
     │  - capabilities                     │
     │  - serverInfo                       │
     │                                     │
     │  3. notifications/initialized       │
     │  ──────────────────────────────────► │
     │                                     │
     │  ✅ Connection Established!         │
     │                                     │
```

### Initialization Exchange:

**Client → Server (Initialize Request):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "elicitation": {}
    },
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  }
}
```

**Server → Client (Initialize Response):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": {}
    },
    "serverInfo": {
      "name": "example-server",
      "version": "1.0.0"
    }
  }
}
```

### Initialization Process တွင် အရေးကြီးသော ရည်ရွယ်ချက်များ:

1. **Protocol Version Negotiation** — Client နှင့် Server compatible version ရှိမရှိ စစ်ဆေးခြင်း
2. **Capability Discovery** — နှစ်ဖက်စလုံး support လုပ်နိုင်တဲ့ features (tools, resources, prompts) ကို ကြေညာခြင်း
3. **Identity Exchange** — Debugging နှင့် compatibility အတွက် identification information ဖလှယ်ခြင်း

---

## 📡 Client Primitives

MCP Server ကနေ Client ကို request လုပ်နိုင်တဲ့ primitives တွေလည်း ရှိပါတယ်:

| Primitive | ရှင်းလင်းချက် |
|---|---|
| **Sampling** | Server ကနေ Client ရဲ့ AI application ဆီ language model completion ကို request လုပ်ခြင်း (`sampling/complete`) |
| **Elicitation** | Server ကနေ user ဆီ additional information သို့မဟုတ် confirmation request လုပ်ခြင်း (`elicitation/request`) |
| **Logging** | Server ကနေ Client ဆီ debugging/monitoring log messages ပို့ခြင်း |

---

## 🔔 Notifications (အသိပေးချက်များ)

Notifications ဟာ real-time updates တွေကို enable လုပ်ပေးပါတယ်:

- **JSON-RPC 2.0 notification messages** (response မလိုပါ)
- `id` field မပါဝင်ပါ
- Server ရဲ့ internal state ပြောင်းလဲမှုပေါ် မူတည်ပြီး ပို့ပါတယ်

**Notifications များ အရေးကြီးတဲ့ အကြောင်းအရင်းများ:**
1. **Dynamic Environments** — Tools, resources တွေ ပေါ်လာခြင်း/ပျောက်သွားခြင်း
2. **Efficiency** — Polling မလိုပဲ changes ကို notify လုပ်ခြင်း
3. **Consistency** — Client တွင် accurate information အမြဲရှိနေခြင်း
4. **Real-time Collaboration** — AI application များ ပြောင်းလဲနေတဲ့ context နဲ့ adapt ဖြစ်ခြင်း

---

## 🔒 Security Considerations

### Server တွင်:
- Tool inputs အားလုံးကို validate လုပ်ရမည်
- Access controls implement လုပ်ရမည်
- Tool invocations ကို rate limit လုပ်ရမည်
- Tool outputs ကို sanitize လုပ်ရမည်
- Resource URIs အားလုံးကို validate လုပ်ရမည်

### Client တွင်:
- Sensitive operations အတွက် user confirmation prompt ပြရမည်
- Tool inputs ကို user ကို ပြရမည် (data exfiltration ကာကွယ်ရန်)
- Tool results ကို LLM ဆီမပို့ခင် validate လုပ်ရမည်
- Tool calls အတွက် timeouts implement လုပ်ရမည်
- Tool usage ကို audit purposes အတွက် log လုပ်ရမည်

---

## ⚠️ Error Handling

MCP တွင် Error reporting mechanism ၂ မျိုးရှိပါတယ်:

### 1. Protocol Errors (JSON-RPC standard errors)

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "error": {
    "code": -32602,
    "message": "Unknown tool: invalid_tool_name"
  }
}
```

| Error Code | ရှင်းလင်းချက် |
|---|---|
| `-32602` | Invalid params (unknown tool, missing arguments) |
| `-32603` | Internal server error |
| `-32002` | Resource not found |

### 2. Tool Execution Errors

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Failed to fetch data: API rate limit exceeded"
      }
    ],
    "isError": true
  }
}
```

---

## 🌟 MCP နဲ့ ဘာတွေ လုပ်နိုင်လဲ?

| Use Case | ရှင်းလင်းချက် |
|---|---|
| 📅 Personal AI Assistant | Google Calendar, Notion နဲ့ ချိတ်ဆက်ပြီး AI assistant အဖြစ် သုံးခြင်း |
| 🎨 Web App Generation | Figma design ကနေ web app တစ်ခုလုံး generate လုပ်ခြင်း |
| 🏢 Enterprise Chatbots | Organization ရဲ့ databases အများကြီးနဲ့ ချိတ်ဆက်ပြီး data analyze လုပ်ခြင်း |
| 🖨️ 3D Design | Blender မှာ 3D design ဖန်တီးပြီး 3D printer နဲ့ ထုတ်ခြင်း |
| 📁 File System Access | Local file system ကို AI ကနေ access လုပ်ခြင်း |
| 🗄️ Database Queries | Database queries ကို AI ကနေ directly run ခြင်း |

---

## 🛠️ MCP SDKs

MCP ကို programming language အမျိုးမျိုးနဲ့ implement လုပ်နိုင်ပါတယ်:

- **TypeScript/JavaScript SDK**
- **Python SDK**
- **Other language SDKs**

---

## 🔗 MCP Ecosystem

| Project | ရှင်းလင်းချက် |
|---|---|
| [MCP Specification](https://modelcontextprotocol.io/specification/latest) | Protocol ရဲ့ official specification |
| [MCP SDKs](https://modelcontextprotocol.io/docs/sdk) | Language-specific SDK implementations |
| [MCP Inspector](https://github.com/modelcontextprotocol/inspector) | MCP Server development/testing tool |
| [MCP Reference Servers](https://github.com/modelcontextprotocol/servers) | Reference server implementations |

---

## 📋 အနှစ်ချုပ်

```
MCP = Open-source standard protocol

     ┌─────────────┐
     │   MCP Host   │  (Claude, VS Code, ChatGPT)
     │  (AI App)    │
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  MCP Client  │  (Connection manager)
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  MCP Server  │  (Context provider)
     │              │
     │  ┌─────────┐ │
     │  │  Tools  │ │  → Executable functions
     │  ├─────────┤ │
     │  │Resources│ │  → Data sources
     │  ├─────────┤ │
     │  │ Prompts │ │  → Interaction templates
     │  └─────────┘ │
     └──────────────┘
```

**MCP ဟာ AI application များကို external system များနဲ့ standardized ပုံစံတစ်မျိုးတည်းနဲ့ ချိတ်ဆက်ပေးတဲ့ protocol ဖြစ်ပြီး — Tools, Resources, Prompts ဆိုတဲ့ core primitives ၃ ခုကို အခြေခံပြီး JSON-RPC 2.0 protocol ပေါ်မှာ အလုပ်လုပ်ပါတယ်။**

---

> 📖 **ထပ်မံလေ့လာရန်:** [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
